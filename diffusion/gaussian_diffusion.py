# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import numpy as np
import torch
import torch as th
from copy import deepcopy
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood, geodesic_distance
from utils.rotation_conversions import rotation_6d_to_matrix_safe
import torch.nn.functional as F

from scipy.interpolate import BSpline

def generate_matrices_for_length(L, K=16, degree=3):
    """Build spline fitting and reconstruction matrices for a sequence."""
    kv = np.concatenate(([0] * degree,
                         np.linspace(0, 1, K - degree + 1),
                         [1] * degree))
    t_points = np.linspace(0, 1, L)

    B = np.zeros((L, K))
    for i in range(K):
        coeffs = np.zeros(K)
        coeffs[i] = 1.0
        spl = BSpline(kv, coeffs, degree)
        B[:, i] = spl(t_points)

    # Ridge regularization stabilizes the least-squares inverse.
    B_t = B.T
    A = np.linalg.inv(B_t @ B + 1e-6 * np.eye(K)) @ B_t

    return A, B

class SplineBottleneck:
    def __init__(self, K=16):
        super().__init__()
        self.K = K
        self.matrix_cache = {}

    def get_matrices(self, L, device):
        if L not in self.matrix_cache:
            A_np, B_np = generate_matrices_for_length(L, self.K)
            A = torch.from_numpy(A_np).float().to(device)
            B = torch.from_numpy(B_np).float().to(device)
            self.matrix_cache[L] = (A, B)
        return self.matrix_cache[L]

    def forward(self, x, lengths, actual_joints):
        bs, T, J, C = x.shape
        out = torch.zeros_like(x)

        for i in range(bs):
            L = int(lengths[i])
            J = int(actual_joints[i])
            if L <= self.K:
                out[i, :L] = x[i, :L]
                continue

            A, B = self.get_matrices(L, x.device)
            x_slice = x[i, :L, :J, :].view(L, -1)
            control_points = torch.mm(A, x_slice)
            smoothed_slice = torch.mm(B, control_points)
            out[i, :L, :J, :] = smoothed_slice.view(L, J, C)

        return out

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        lambda_fs=0.,
        lambda_geo=0.
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.lambda_fs = lambda_fs
        self.lambda_geo = lambda_geo

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = lambda a, b: (a - b) ** 2  # th.nn.MSELoss(reduction='none')  # must be None for handling mask later on.

        self.spline = SplineBottleneck(K=16)
    def masked_l2(self, a, b, mask):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming mask.shape == bs, 1, 1, seqlen
        loss = self.l2_loss(a, b)
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def velocity_loss_with_tail_mask(self, a, b, temp_mask, spat_mask, lengths, n_joints):
        """Compute masked velocity loss while treating the final frame as static."""
        vel_a = th.diff(a, dim=3, append=a[..., -1:])
        vel_b = th.diff(b, dim=3, append=b[..., -1:])

        shifted_mask = th.cat([temp_mask[..., 1:], th.zeros_like(temp_mask[..., :1])], dim=3)
        v_temp_mask = temp_mask * shifted_mask

        v_loss = self.l2_loss(vel_a, vel_b)
        masked_v_loss = v_loss * v_temp_mask.float() * spat_mask.float().transpose(1, 3)

        return sum_flat(masked_v_loss) / ((lengths - 1) * n_joints * a.size(2))

    def temporal_spatial_masked_l2(self, a, b, temp_mask, spat_mask, lengths, n_joints):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming temp_mask.shape == bs, 1, 1, seqlen
        # assuming spat_mask.shape == bs, 1, 1, max_joints

        loss = self.l2_loss(a, b)
        temp_masked_loss = loss * temp_mask.float()
        spat_temp_masked_loss = (temp_masked_loss * spat_mask.float().transpose(1,3))
        loss = sum_flat(spat_temp_masked_loss)  # gives \sigma_euclidean over unmasked elements
        non_zero_elements = lengths * n_joints * a.size(2)
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def relative_spatial_masked_l2(self, a, temp_mask, spat_mask, lengths, n_joints):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming temp_mask.shape == bs, 1, 1, seqlen
        # assuming spat_mask.shape == bs, 1, 1, max_joints

        loss = a
        temp_masked_loss = loss * temp_mask.float()
        spat_temp_masked_loss = (temp_masked_loss * spat_mask.float().transpose(1,3))
        loss = sum_flat(spat_temp_masked_loss)  # gives \sigma_euclidean over unmasked elements
        non_zero_elements = lengths * n_joints * a.size(2)
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def quat_to_mat(self, qs):
        r = qs[..., 0]
        i = qs[..., 1]
        j = qs[..., 2]
        k = qs[..., 3]
        two_s = 2.0 / (qs * qs).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        rotations = o.reshape(qs.shape[:-1] + (3, 3))
        return rotations

    def geodesic_loss(self, a, b, temp_mask, spat_mask, lengths, n_joints):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming temp_mask.shape == bs, 1, 1, seqlen
        # assuming spat_mask.shape == bs, 1, 1, max_joints
        rots_target = rotation_6d_to_matrix_safe(a.permute(0, 3, 1, 2)[..., 3:9])
        rots_pred = rotation_6d_to_matrix_safe(b.permute(0, 3, 1, 2)[..., 3:9])
        loss = geodesic_distance(rots_pred, rots_target).permute(0, 2, 3, 1)
        temp_masked_loss = loss * temp_mask.float()
        spat_temp_masked_loss = (temp_masked_loss * spat_mask.float().transpose(1,3))
        loss = sum_flat(spat_temp_masked_loss)  # gives \sigma_euclidean over unmasked elements
        non_zero_elements = lengths * n_joints
        loss_val = loss / non_zero_elements
        return loss_val

    def foot_sliding_loss(self, a, b, temp_mask, relative=True):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming temp_mask.shape == bs, 1, 1, seqlen
        # assuming spat_mask.shape == bs, 1, 1, max_joints

        # gt foot contact. Multiplied by temporal mask to zero out the last frame fc. Spatial mask is not needed since irrelevant joints fc is already zero
        fc = a[..., 12, :-1] * temp_mask[..., 0, 1:] != 0   # (bs, J, seqlen-1)

        # compute root relative velocity from ric positions
        vel_tgt = (a[..., :3, 1:] - a[..., :3, :-1]) # (bs, J, 3, seqlen-1)
        vel_pred = (b[..., :3, 1:] - b[..., :3, :-1]) # (bs, J, 3, seqlen-1)
        if relative:
            loss_term = self.l2_loss(vel_tgt, vel_pred).sum(dim=-2).sqrt() * fc
        else:
            loss_term = (vel_pred**2).sum(dim=-2).sqrt() * fc

        loss = sum_flat(loss_term)  # gives \sigma_euclidean over unmasked elements
        non_zero_elements = 1 + torch.count_nonzero(fc, dim = (-1, -2)) # +1 to rement division by zero
        loss_val = loss / non_zero_elements
        return loss_val

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, get_layer_activation=-1, guidance_scale=1.0
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        if guidance_scale != 1.0 and model_kwargs:
            if get_layer_activation > -1 and get_layer_activation < model.model.num_layers :
                model_output_cond, activations = model(x, self._scale_timesteps(t), get_layer_activation=get_layer_activation, **model_kwargs)
            else:
                model_output_cond = model(x, self._scale_timesteps(t), **model_kwargs)

            model_kwargs_uncond = deepcopy(model_kwargs)
            model_kwargs_uncond["y"]["video_dino"] = th.zeros_like(model_kwargs_uncond["y"]["video_dino"])
            if get_layer_activation > -1 and get_layer_activation < model.model.num_layers :
                model_output_uncond, _ = model(x, self._scale_timesteps(t), get_layer_activation=get_layer_activation, **model_kwargs_uncond)
            else:
                model_output_uncond = model(x, self._scale_timesteps(t), **model_kwargs_uncond)
            cfg_noise = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
            dims = list(range(1, cfg_noise.ndim))

            std_cond = model_output_cond.std(dim=dims, keepdim=True)
            std_cfg = cfg_noise.std(dim=dims, keepdim=True)

            # Match the guided prediction magnitude to the conditional output.
            factor = std_cond / (std_cfg + 1e-8)

            rescale_phi = 0.7
            cfg_noise_rescaled = cfg_noise * factor

            model_output = rescale_phi * cfg_noise_rescaled + (1 - rescale_phi) * cfg_noise
        else:
            if get_layer_activation > -1 and get_layer_activation < model.model.num_layers :
                model_output, activations = model(x, self._scale_timesteps(t), get_layer_activation=get_layer_activation, **model_kwargs)
            else:
                model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys():
            inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
            assert self.model_mean_type == ModelMeanType.START_X, 'This feature supports only X_start pred for mow!'
            assert model_output.shape == inpainting_mask.shape == inpainted_motion.shape
            model_output = (model_output * ~inpainting_mask) + (inpainted_motion * inpainting_mask)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]


            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        if get_layer_activation > -1 and get_layer_activation < model.model.num_layers:
            return {
                "mean": model_mean,
                "variance": model_variance,
                "log_variance": model_log_variance,
                "pred_xstart": pred_xstart,
                "activations": activations
            }
        else:
            return {
                "mean": model_mean,
                "variance": model_variance,
                "log_variance": model_log_variance,
                "pred_xstart": pred_xstart,
            }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        guidance_scale=1.0
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            guidance_scale=guidance_scale
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_dift(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
        get_layer_activation = -1
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            get_layer_activation=get_layer_activation
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        if get_layer_activation > -1:
            return {"sample": sample, "pred_xstart": out["pred_xstart"], "activations": out["activations"]}
        else:
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
        get_activations={"layer": -1, "timestep": -1},
        guidance_scale=1.0
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []
        activations_dict=dict()
        sample_loop = self.p_sample_loop_progressive
        for i, sample in enumerate(sample_loop(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
            guidance_scale=guidance_scale
        )):
            final = sample
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))

        if dump_steps is not None:
            return dump
        return final["sample"]

    def p_sample_single_timestep(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        init_image=None,
        randomize_class=False,
        const_noise=False,
        get_activations={"layer": -1, "timestep": -1},
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param t: diffusion timestep.
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        activations_dict=dict()
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is None:
            noise = th.randn(*shape, device=device)
        img = th.randn(*shape, device=device)
        t = get_activations["timestep"]
        t_tnsr = th.tensor([t] * shape[0], device=device)
        if init_image is not None:
            # randomize the initial image to timestep t for dift extractrion
            img = self.q_sample(init_image, t_tnsr, noise)
        if randomize_class and 'y' in model_kwargs:
            model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                            size=model_kwargs['y'].shape,
                                            device=model_kwargs['y'].device)
        with th.no_grad():
            out = self.p_sample_dift(
                model,
                img,
                t_tnsr,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                const_noise=const_noise,
                get_layer_activation=get_activations["layer"]
            )

            if get_activations["timestep"] == t:
                activations_dict[t] = out["activations"]

        if get_activations["timestep"] == t:
            return out["sample"], activations_dict
        return out["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
        get_activations=None, # dummy
        guidance_scale=1.0
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                    guidance_scale=guidance_scale
                )
                yield out
                img = out["sample"]

    def p_sample_loop_for_dift(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        const_noise=False,
        get_activations={"layer": -1, "timestep": -1}

    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            noise = th.randn(*shape, device=device)
        img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            get_layer_activation = -1
            if get_activations["timestep"] == i:
                get_layer_activation = get_activations["layer"]
            t = th.tensor([i] * shape[0], device=device)
            if init_image is not None:
                # randomize the initial image to timestep t for dift extractrion
                img = self.q_sample(init_image, t, noise)

            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.p_sample_dift
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                    get_layer_activation = get_layer_activation
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        """
        Sample x_{t-1} from the model using Pseudo Linear Multistep.

        Same usage as p_sample().
        """
        if not int(order) or not 1 <= order <= 4:
            raise ValueError('order is invalid (should be int from 1-4).')

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                else:
                    out = out_orig

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            # Pseudo Improved Euler
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime
        else:
            # Pseudo Linear Multistep (Adams-Bashforth)
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] + 5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] + 37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError('cur_order is invalid.')
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "old_eps": old_eps}

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Generate samples from the model using Pseudo Linear Multistep.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from each
        timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def rotation_6d_to_matrix_torch(self, cont6d):
        """Convert 6D rotations to 3x3 rotation matrices."""
        x_raw = cont6d[..., 0:3]
        y_raw = cont6d[..., 3:6]

        x = F.normalize(x_raw, p=2, dim=-1)
        z = torch.cross(x, y_raw, dim=-1)
        z = F.normalize(z, p=2, dim=-1)
        y = torch.cross(z, x, dim=-1)

        matrix = torch.stack((x, y, z), dim=-1)
        return matrix

    def recover_root_quat_and_pos_torch(self, data):
        """Recover root rotations and positions from motion features."""
        r_rot_mat = self.rotation_6d_to_matrix_torch(data[..., 3:9])

        r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device, dtype=data.dtype)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
        r_pos = torch.matmul(r_rot_mat, r_pos.unsqueeze(-1)).squeeze(-1)
        r_pos = torch.cumsum(r_pos, dim=-2)
        r_pos[..., 1] = data[..., 1]
        return r_rot_mat, r_pos

    def recover_from_ric_torch(self, data):
        """Recover global joint positions from root-relative coordinates."""
        r_rot_mat, r_pos = self.recover_root_quat_and_pos_torch(data[..., 0, :])
        positions = data[..., 1:, :3]

        B, F, J_minus_1, _ = positions.shape
        r_rot_mat_expanded = r_rot_mat.unsqueeze(-3).expand(B, F, J_minus_1, 3, 3)
        positions = torch.matmul(r_rot_mat_expanded, positions.unsqueeze(-1)).squeeze(-1)

        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]
        full_positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
        return full_positions

    def compute_bone_length_loss(self, pred_joints, gt_first_frame, relations):
        """
        Args:
            pred_joints: (bs, F, J, 3)
            gt_first_frame: (bs, J, 3)
            relations: (bs, J, J)
        Returns:
            loss: per-sample bone-length loss with shape ``(bs,)``.
        """
        bs, F, J, _ = pred_joints.shape
        mask = (relations == 2)

        gt_diffs = gt_first_frame.unsqueeze(1) - gt_first_frame.unsqueeze(2)
        target_lengths = torch.sqrt(torch.sum(gt_diffs**2, dim=-1) + 1e-8)

        pred_diffs = pred_joints.unsqueeze(2) - pred_joints.unsqueeze(3)
        curr_lengths = torch.sqrt(torch.sum(pred_diffs**2, dim=-1) + 1e-8)

        target_lengths = target_lengths.unsqueeze(1)
        sq_err = (curr_lengths - target_lengths) ** 2
        expanded_mask = mask.unsqueeze(1).expand(-1, F, -1, -1)
        masked_err = sq_err * expanded_mask
        bones_per_sample = mask.sum(dim=(1, 2)) * F
        sample_loss = masked_err.sum(dim=(1, 2, 3)) / (bones_per_sample + 1e-8)
        return sample_loss


    def temporal_spatial_masked_global_l2(self, a, b, temp_mask, spat_mask, lengths, n_joints):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming temp_mask.shape == bs, 1, 1, seqlen
        # assuming spat_mask.shape == bs, 1, 1, max_joints
        loss = self.l2_loss(a, b)
        temp_masked_loss = loss * temp_mask.float()
        spat_temp_masked_loss = (temp_masked_loss * spat_mask.float().transpose(1,3))
        loss = sum_flat(spat_temp_masked_loss)  # gives \sigma_euclidean over unmasked elements
        non_zero_elements = lengths * n_joints * a.size(2)
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val


    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        mask = model_kwargs['y']['lengths_mask']
        lengths = model_kwargs['y']['lengths']
        actual_joints = model_kwargs['y']['n_joints']
        joints_mask = model_kwargs['y']['joints_mask'][:, :, :, 1, 1:]
        mean = model_kwargs['y']['mean'][..., None]
        std = model_kwargs['y']['std'][..., None]
        tpos_first_frame = model_kwargs['y']['tpos_first_frame']
        temp_mask = model_kwargs['y']['mask']

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape  # [bs, njoints, nfeats, nframes]
            # get ric positions befor denorm to care equally for all topologies
            terms["l_simple"] = self.temporal_spatial_masked_l2(target, model_output, mask, joints_mask, lengths, actual_joints)
            terms["loss"] = torch.zeros_like(terms["l_simple"])
            terms["loss"] = terms["loss"] + terms["l_simple"]

            if self.lambda_geo > 0.:
                terms["geodesic_loss"] = self.geodesic_loss(target, model_output, mask, joints_mask, lengths, actual_joints)
                terms["loss"] = terms["loss"] + self.lambda_geo * terms["geodesic_loss"]
            if self.lambda_fs > 0.:
                terms["foot_sliding_loss"] = self.foot_sliding_loss(target, model_output, mask, relative=True)
                terms["loss"] = terms["loss"] + self.lambda_fs * terms["foot_sliding_loss"]

        else:
            raise NotImplementedError(self.loss_type)

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
