import torch
import torch.nn as nn
from typing import Optional, Union, Callable, Tuple
from torch import Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from einops import rearrange
CUDA_LAUNCH_BLOCKING=1

class SpatialRoPE2D(nn.Module):
    def __init__(self, dim, max_h=64, max_w=64):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2

        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, h, w, device):
        pos_h = torch.arange(h, device=device).type_as(self.inv_freq)
        pos_w = torch.arange(w, device=device).type_as(self.inv_freq)

        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq)
        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq)

        emb_h = freqs_h[:, None, :].expand(h, w, -1)
        emb_w = freqs_w[None, :, :].expand(h, w, -1)

        freqs = torch.cat([emb_h, emb_w], dim=-1)

        # Duplicate frequencies to match rotate_half's paired layout.
        emb = torch.cat((freqs, freqs), dim=-1)

        emb = rearrange(emb, 'h w d -> 1 (h w) 1 d')
        return emb.cos(), emb.sin()

class SpatialFlashAttention2D(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.rope_2d = SpatialRoPE2D(self.head_dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """Apply spatial attention to ``(batch, frames, height, width, dim)`` features."""
        bs, f, h, w, dim = x.shape

        x_in = rearrange(x, 'b f h w d -> (b f) (h w) d')

        qkv = self.qkv(x_in).reshape(-1, h * w, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        cos, sin = self.rope_2d(h, w, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(out, 'bf h n d -> bf n (h d)')
        out = self.proj(out)
        x_out = self.norm(x_in + out)

        return rearrange(x_out, '(b f) (h w) d -> b f h w d', b=bs, f=f, h=h, w=w)

class TemporalRoPE(nn.Module):
    def __init__(self, dim, max_seq_len=64):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DINOWithTemporalAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rope = TemporalRoPE(self.head_dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """Apply temporal attention independently at each spatial location."""
        bs, f, h, w, dim = x.shape

        x_in = rearrange(x, 'b f h w d -> (b h w) f d', b=bs, h=h, w=w)

        qkv = self.qkv(x_in).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b f (h d) -> b f h d', h=self.num_heads), qkv)

        cos, sin = self.rope(f, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v.transpose(1, 2)).transpose(1, 2)
        out = rearrange(out, 'b f h d -> b f (h d)')

        out = self.proj(out)
        x_out = self.norm(x_in + out)

        x_out = rearrange(x_out, '(b h w) f d -> b f h w d', b=bs, h=h, w=w)

        return x_out

class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model, dropout, nheads):
        super().__init__()

        self.nheads = nheads

        self.att_size = att_size = d_model // nheads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(d_model, nheads * att_size)
        self.linear_k = nn.Linear(d_model, nheads * att_size)
        self.linear_v = nn.Linear(d_model, nheads * att_size)
        self.dropout = nn.Dropout(dropout)

        self.output_layer = nn.Linear(nheads * att_size, d_model)

    def forward(
        self,
        q,
        k,
        v,
        query_hop_emb,
        query_edge_emb,
        key_hop_emb,
        key_edge_emb,
        value_hop_emb,
        value_edge_emb,
        distance,
        edge_attr,
        mask=None,
    ):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.nheads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.nheads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.nheads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2)  # [b, h, k_len, d_k]

        sequence_length = v.shape[2]
        num_hop_types = query_hop_emb.shape[0]
        num_edge_types = query_edge_emb.shape[0]

        query_hop_emb = query_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        query_edge_emb = query_edge_emb.view(
            1, -1, self.nheads, self.att_size
        ).transpose(1, 2)
        key_hop_emb = key_hop_emb.view(
            1, num_hop_types, self.nheads, self.att_size
        ).transpose(1, 2)
        key_edge_emb = key_edge_emb.view(
            1, num_edge_types, self.nheads, self.att_size
        ).transpose(1, 2)

        query_hop = torch.matmul(q, query_hop_emb.transpose(2, 3))
        query_hop = torch.gather(
            query_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )
        query_edge = torch.matmul(q, query_edge_emb.transpose(2, 3))
        query_edge = torch.gather(
            query_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )

        key_hop = torch.matmul(k, key_hop_emb.transpose(2, 3))
        key_hop = torch.gather(
            key_hop, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )
        key_edge = torch.matmul(k, key_edge_emb.transpose(2, 3))
        key_edge = torch.gather(
            key_edge, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1)
        )

        spatial_bias = (query_hop + key_hop)
        edge_bais = (query_edge + key_edge)

        x = torch.matmul(q, k.transpose(2, 3)) + spatial_bias + edge_bais

        x = x * self.scale

        if mask is not None:
            x = x + mask

        x = torch.softmax(x, dim=3)
        x = self.dropout(x)
        if value_hop_emb is not None:
            value_hop_emb = value_hop_emb.view(
                1, num_hop_types, self.nheads, self.att_size
            ).transpose(1, 2)
            value_edge_emb = value_edge_emb.view(
                1, num_edge_types, self.nheads, self.att_size
            ).transpose(1, 2)

            value_hop_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_hop_types),
                device=value_hop_emb.device,
            )
            value_hop_att = torch.scatter_add(
                value_hop_att, 3, distance.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
            value_edge_att = torch.zeros(
                (batch_size, self.nheads, sequence_length, num_edge_types),
                device=value_hop_emb.device,
            )
            value_edge_att = torch.scatter_add(
                value_edge_att, 3, edge_attr.unsqueeze(1).repeat(1, self.nheads, 1, 1), x
            )
        x = torch.matmul(x, v)
        if value_hop_emb is not None:
            x = x + torch.matmul(value_hop_att, value_hop_emb) + torch.matmul(value_edge_att, value_edge_emb)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.nheads * d_v)

        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x

class GraphMotionDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None, max_path_len=5, value_emb=False):
        super().__init__(decoder_layer, num_layers, norm)

        self.d_model = decoder_layer.d_model
        self.topology_key_emb = nn.Embedding(max_path_len + 1, self.d_model)
        self.edge_key_emb = nn.Embedding(6, self.d_model)
        self.topology_query_emb = nn.Embedding(max_path_len + 1, self.d_model)
        self.edge_query_emb = nn.Embedding(6, self.d_model)
        self.value_emb_flag = value_emb
        if value_emb:
            self.topology_value_emb = nn.Embedding(max_path_len + 1, self.d_model)
            self.edge_value_emb = nn.Embedding(6, self.d_model)



    def forward(self, tgt: Tensor, timesteps_embs: Tensor, memory: Tensor, spatial_mask:  Optional[Tensor] = None,
                temporal_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, y=None, get_layer_activation=-1, input_video=None) -> Union[Tensor , Tuple[Tensor, dict]]:
        topology_rel = y['graph_dist'].long().to(tgt.device)
        edge_rel = y['joints_relations'].long().to(tgt.device)
        output = tgt
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            activations=dict()
        for layer_ind, mod in enumerate(self.layers):
            edge_value_emb = None
            topology_value_emb = None
            if self.value_emb_flag:
                edge_value_emb = self.edge_value_emb
                topology_value_emb = self.topology_value_emb
            output, input_video = mod(
                    output, timesteps_embs, topology_rel, edge_rel, self.edge_key_emb, self.edge_query_emb, edge_value_emb, self.topology_key_emb, self.topology_query_emb, topology_value_emb, spatial_mask, temporal_mask,
                    tgt_key_padding_mask, memory_key_padding_mask, y, input_video)
            if layer_ind == get_layer_activation:
                activations[layer_ind] = output.clone()
        if self.norm is not None:
            output = self.norm(output)
        if get_layer_activation > -1 and get_layer_activation < self.num_layers:
            return output, activations
        return output

class GraphMotionDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.d_model= d_model
        self.heads = nhead
        self.spatial_attn = GraphMultiHeadAttention(d_model = d_model, nheads = nhead, dropout=dropout)
        self.temporal_attn = MultiheadAttention(self.d_model, nhead, dropout=dropout)
        self.embed_timesteps = nn.Linear(d_model, d_model)


        self.dino_proj = nn.Linear(768, d_model)
        self.video_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.reverse_video_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.rest_pose_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.video_dropout = nn.Dropout(dropout)
        self.dino_dropout = nn.Dropout(dropout)
        self.skeleton_dropout = nn.Dropout(dropout)
        self.norm_video = nn.LayerNorm(d_model)
        self.norm_dino = nn.LayerNorm(d_model)
        self.norm_skeleton = nn.LayerNorm(d_model)
        self.norm_skeleton_2 = nn.LayerNorm(d_model)

        self.dino_temporal_attn = DINOWithTemporalAttention(dim=d_model, num_heads=nhead)
        self.dino_spatial_attn = SpatialFlashAttention2D(dim=d_model, num_heads=nhead)

        self.linearup = nn.Linear(d_model, dim_feedforward)
        self.lineardown = nn.Linear(dim_feedforward, d_model)
        self.dropout_ff = nn.Dropout(dropout)

    def _spatial_mha_block(self, x: Tensor, topology_rel: Optional[Tensor], edge_rel: Optional[Tensor], edge_key_emb, edge_query_emb, edge_value_emb,
        topology_key_emb, topology_query_emb, topology_value_emb, attn_mask: Optional[Tensor],  key_padding_mask: Optional[Tensor], y = None) -> Tensor:
        frames, bs, njoints, feature_len = x.shape
        x = x.view(frames * bs, njoints, feature_len)
        topology_rel = topology_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, njoints, njoints)
        edge_rel = edge_rel.unsqueeze(0).repeat(frames, 1, 1, 1).view(-1, njoints, njoints)

        attn_output = self.spatial_attn(x, x, x, topology_query_emb.weight, edge_query_emb.weight, topology_key_emb.weight, edge_key_emb.weight, None if topology_value_emb is None else topology_value_emb.weight,
        None if edge_value_emb is None else edge_value_emb.weight, topology_rel, edge_rel, attn_mask)
        attn_output = attn_output.reshape(frames, bs, njoints, feature_len)
        return self.dropout1(attn_output)


    def _temporal_mha_block_sin_joint(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        frames, bs, njoints, feats= x.size()
        x = x.view(frames, bs * njoints, feats)
        output_attn, output_scores = self.temporal_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        output_attn = output_attn.view(frames, bs ,njoints, feats)
        return self.dropout2(output_attn)

    def _reverse_video_cross_attn_block(self, x: Tensor, video_features: Tensor,
                                    video_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Update video tokens by attending to the skeletal representation."""
        frames, bs, njoints, feats = x.size()
        orig_shape = video_features.shape

        video_feat = video_features.flatten(3, 4).permute(0, 1, 3, 2)

        query = video_feat.permute(2, 0, 1, 3).reshape(1369, bs * frames, feats)

        key = value = x.permute(2, 1, 0, 3).reshape(njoints, bs * frames, feats)

        if video_key_padding_mask is not None:
            rev_kp_mask = video_key_padding_mask.view(bs * frames).unsqueeze(1).repeat(1, njoints)
        else:
            rev_kp_mask = None

        attn_output, _ = self.reverse_video_cross_attn(
            query, key, value,
            key_padding_mask=rev_kp_mask
        )

        attn_output = attn_output.view(1369, bs, frames, feats).permute(1, 2, 0, 3)
        video_feat_out = attn_output.permute(0, 1, 3, 2).reshape(orig_shape)

        return self.skeleton_dropout(video_feat_out)

    def _video_cross_attn_block(self, x: Tensor, video_features: Tensor,
                            video_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Update skeletal tokens by attending to per-frame video tokens."""
        frames, bs, njoints, feats = x.size()

        video_feat = video_features.flatten(2, 3)

        query = x.permute(2, 1, 0, 3).reshape(njoints, bs * frames, feats)
        key = value = video_feat.permute(2, 0, 1, 3).reshape(1369, bs * frames, feats)

        if video_key_padding_mask is not None:
            video_kp_mask = video_key_padding_mask.repeat_interleave(1369, dim=0)
        else:
            video_kp_mask = None

        output_attn, _ = self.video_cross_attn(
            query, key, value,
            attn_mask=None,
            key_padding_mask=video_kp_mask
        )

        output_attn = output_attn.view(njoints, bs, frames, feats).permute(2, 1, 0, 3)

        return self.video_dropout(output_attn)

    def _rest_pose_cross_attn_block(self, x: Tensor, rest_pose_features: Tensor, tpos_first_frame: Tensor) -> Tensor:
        """Inject per-joint appearance features into skeletal tokens."""
        frames, bs, njoints, feats = x.size()

        video_feat = rest_pose_features
        video_feat = self.dino_proj(video_feat)

        query = x.reshape(frames, bs * njoints, feats)
        key = value = video_feat.reshape(1, bs * njoints, feats)

        output_attn, _ = self.rest_pose_cross_attn(
            query, key, value,
            attn_mask=None,
            key_padding_mask=None,
        )

        output_attn = output_attn.view(frames, bs, njoints, feats)
        return self.dino_dropout(output_attn)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def _ff_block_2(self, x: Tensor) -> Tensor:
        x = self.lineardown(self.dropout(self.activation(self.linearup(x))))
        return self.dropout_ff(x)

    def forward(self,
        tgt: Tensor,
        timesteps_emb: Tensor,
        topology_rel: Tensor,
        edge_rel: Tensor,
        edge_key_emb,
        edge_query_emb,
        edge_value_emb,
        topo_key_emb,
        topo_query_emb,
        topo_value_emb,
        spatial_mask: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        y = None,
        input_video=None) -> Tensor:
        x = tgt
        bs = x.shape[1]
        x = x + self.embed_timesteps(timesteps_emb).view(1, bs, 1, self.d_model)
        spatial_attn_output = self._spatial_mha_block(x, topology_rel, edge_rel, edge_key_emb, edge_query_emb, edge_value_emb,
        topo_key_emb, topo_query_emb, topo_value_emb, spatial_mask, tgt_key_padding_mask, y)
        x = self.norm1(x + spatial_attn_output)
        x = self.norm2(x + self._temporal_mha_block_sin_joint(x, temporal_mask, tgt_key_padding_mask))
        input_video = self.dino_spatial_attn(input_video)
        input_video = self.dino_temporal_attn(input_video)
        video_attn_output = self._video_cross_attn_block(
            x, input_video, tgt_key_padding_mask
        )
        x = self.norm_video(x + video_attn_output)
        skeleton_attn_output = self._reverse_video_cross_attn_block(
            x, input_video, tgt_key_padding_mask
        )
        input_video = self.norm_skeleton(skeleton_attn_output + input_video)
        dino_attn_output = self._rest_pose_cross_attn_block(
            x, y['joints_names_embs'].to(x.device), None
        )
        x = self.norm_dino(x + dino_attn_output)
        x = self.norm3(x + self._ff_block(x))

        input_video = self.norm_skeleton_2(input_video + self._ff_block_2(input_video))
        return x, input_video
