import torch
torch.cuda.empty_cache()
import torch.nn as nn
from model.motion_transformer import GraphMotionDecoderLayer, GraphMotionDecoder
import timm


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

class SkelMo(nn.Module):
    def __init__(self, max_joints, feature_len,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", root_input_feats=12,
                 **kargs):
        super().__init__()

        self.max_joints = max_joints
        self.feature_len = feature_len
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = self.feature_len
        self.root_input_feats = root_input_feats
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.value_emb=kargs.get('value_emb', False)
        self.video_feature_mode = kargs.get('video_feature_mode', 'rgb')
        self.video_track_dim = int(kargs.get('video_track_dim', 776))
        if self.video_feature_mode not in {'rgb', 'dense_dino', 'joint_tracks'}:
            raise ValueError(f"Unsupported video feature mode: {self.video_feature_mode}")
        self.input_process = InputProcess(self.input_feats, self.root_input_feats, self.latent_dim)
        # Distinguish the structural NULL slot from observed motion frames.
        self.visual_token_type_embedding = nn.Embedding(2, self.latent_dim)
        nn.init.zeros_(self.visual_token_type_embedding.weight)

        print("Graph transformer init")
        seqTransDecoderLayer = GraphMotionDecoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation,
                                                            compact_video=self.video_feature_mode == 'joint_tracks')
        self.seqTransDecoder = GraphMotionDecoder(seqTransDecoderLayer,
                                                        num_layers=self.num_layers, value_emb=self.value_emb)


        self.output_process = OutputProcess(self.feature_len, self.root_input_feats, self.max_joints, self.latent_dim)

        if self.video_feature_mode == 'joint_tracks':
            self.video_proj = nn.Sequential(
                nn.LayerNorm(self.video_track_dim),
                nn.Linear(self.video_track_dim, self.latent_dim),
            )
            self.visual_model = None
        else:
            self.video_proj = nn.Linear(768, self.latent_dim)
            self.visual_model = None
            if self.video_feature_mode == 'rgb':
                self.visual_model = timm.create_model(
                    "vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
                )
                # DINOv2 supplies frozen visual features throughout training.
                self.visual_model.eval()
                for param in self.visual_model.parameters():
                    param.requires_grad = False

    def forward(self, x, timesteps, get_layer_activation=-1, y=None, inference=None):
        """Denoise ``x`` with shape ``[batch, joints, features, frames]``."""

        joints_mask = y['joints_mask'].to(x.device)
        temp_mask = y['mask'].to(x.device)
        tpos_first_frame = y['tpos_first_frame'].to(x.device).unsqueeze(0)
        video_dino = y['video_dino'].to(x.device)

        if self.video_feature_mode == 'joint_tracks':
            if video_dino.ndim != 4 or video_dino.shape[-1] != self.video_track_dim:
                raise ValueError(
                    "joint_tracks expects video features [B,F,J,D] with "
                    f"D={self.video_track_dim}; received {tuple(video_dino.shape)}"
                )
            video_dino = self.video_proj(video_dino.float())
            frame_ids = torch.arange(video_dino.shape[1], device=x.device)[None, :, None]
            joint_ids = torch.arange(video_dino.shape[2], device=x.device)[None, None, :]
            valid = (
                frame_ids < (y['lengths'].to(x.device)[:, None, None] + 1)
            ) & (
                joint_ids < y['n_joints'].to(x.device)[:, None, None]
            )
            token_types = torch.ones(video_dino.shape[1], dtype=torch.long, device=x.device)
            # Slot zero is a learned-type null aligned with numeric REST geometry.
            token_types[0] = 0
            video_dino = video_dino + self.visual_token_type_embedding(token_types)[None, :, None]
            video_dino = video_dino.masked_fill(~valid[..., None], 0)
        else:
            if inference is None:
                inference = self.video_feature_mode == 'dense_dino' or video_dino.ndim == 4
            if inference:
                B, F, token_count, channels = video_dino.shape
                features = video_dino.reshape(B * F, token_count, channels)
            else:
                if self.visual_model is None:
                    raise RuntimeError("Raw RGB input requested without a visual backbone.")
                B, F, C, H, W = video_dino.shape
                features = self.visual_model.forward_features(
                    video_dino.reshape(-1, C, H, W).float()
                )

            patch_feats = features[:, 1:, :]
            patch_count = patch_feats.shape[1]
            side = int(patch_count ** 0.5)
            if side * side != patch_count:
                raise ValueError(f"DINO patch count is not square: {patch_count}")
            video_dino = patch_feats.reshape(B, F, side, side, -1)
            video_dino = self.video_proj(video_dino)
            token_types = torch.ones(F, dtype=torch.long, device=x.device)
            # Slot zero is a learned-type null aligned with numeric REST geometry.
            token_types[0] = 0
            video_dino = video_dino + self.visual_token_type_embedding(
                token_types
            )[None, :, None, None]

        bs, njoints, nfeats, nframes = x.shape
        timesteps_emb = create_sin_embedding(timesteps.view(1, -1, 1), self.latent_dim)[0]
        x = self.input_process(x, tpos_first_frame, y['joints_names_embs'].to(x.device), y['crop_start_ind'])

        spatial_mask = 1.0 - joints_mask[:, 0, 0, 1:, 1:]
        spatial_mask = spatial_mask.unsqueeze(1).unsqueeze(1).repeat(1, nframes + 1, self.num_heads, 1, 1).reshape(-1,self.num_heads, njoints, njoints)
        temporal_mask = 1.0 - temp_mask.repeat(1, njoints, self.num_heads, 1, 1).reshape(-1, nframes + 1, nframes + 1).float()
        spatial_mask[spatial_mask == 1.0] = -1e9
        temporal_mask[temporal_mask == 1.0] = -1e9

        output = self.seqTransDecoder(tgt=x, timesteps_embs=timesteps_emb, memory=None, spatial_mask=spatial_mask, temporal_mask = temporal_mask, y=y, get_layer_activation=get_layer_activation, input_video=video_dino)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations = output[1]
            output=output[0]
        output = self.output_process(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output


    def _apply(self, fn):
        super()._apply(fn)
        return self


    def train(self, mode=True):
        super().train(mode)
        if self.visual_model is not None:
            self.visual_model.eval()
        return self


# Backward-compatible name for checkpoints and downstream imports based on AnyTop.
AnyTop = SkelMo

class InputProcess(nn.Module):
    def __init__(self, input_feats, root_input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.root_input_feats = root_input_feats
        self.root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.tpos_root_embedding = nn.Linear(self.root_input_feats, self.latent_dim)
        self.joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.tpos_joint_embedding = nn.Linear(self.input_feats, self.latent_dim)
        self.joints_feature_dropout = nn.Dropout(p=0.1)
        self.feature_embedding = nn.Linear(768, self.latent_dim)

    def forward(self, x, tpos_first_frame, joints_embedded_names, crop_start_ind):
        x = x.permute(3, 0, 1, 2)
        tpos_all_joints_except_root = self.tpos_joint_embedding(tpos_first_frame[:, :, 1:])
        tpos_root_data = self.tpos_root_embedding(tpos_first_frame[:, :, 0:1])
        all_joints_except_root = self.joint_embedding(x[:, :, 1:])
        root_data = self.root_embedding(x[:, :, 0:1])
        tpos_embedded = torch.cat([tpos_root_data, tpos_all_joints_except_root], dim=2)
        x_embedded = torch.cat([root_data, all_joints_except_root], dim=2)

        x = torch.cat([tpos_embedded, x_embedded], dim=0)

        positions = torch.arange(x.shape[0], device=x.device).view(1, -1, 1).repeat(x.shape[1], 1, 1)
        positions[:,1:,:] = positions[:,1:,:] + crop_start_ind.to(x.device).view(-1, 1, 1)
        pos_emb = create_sin_embedding(positions, self.latent_dim)[0]
        return x + pos_emb.unsqueeze(1).unsqueeze(1)

class OutputProcess(nn.Module):
    def __init__(self, feature_len, root_feature_len, max_joints, latent_dim):
        super().__init__()
        self.feature_len = feature_len
        self.max_joints = max_joints
        self.latent_dim = latent_dim
        self.root_feature_len = root_feature_len
        self.root_dembedding = nn.Linear(self.latent_dim, self.root_feature_len)
        self.joint_dembedding = nn.Linear(self.latent_dim, self.feature_len)

    def forward(self, output):
        root_data = self.root_dembedding(output[:, :, 0])
        all_joints = self.joint_dembedding(output[:, :, 1:])
        output = torch.cat([root_data.unsqueeze(2), all_joints], dim=-2)
        output = output.permute(1, 2, 3, 0)[..., 1:]
        return output
