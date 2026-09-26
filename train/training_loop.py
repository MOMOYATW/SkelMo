import functools
import os
import re
from os.path import join as pjoin
from typing import Optional
import blobfile as bf
import torch
from torch.optim import AdamW
from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from sample.generate import main as generate
import copy
from utils.model_util import load_model
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import torch.distributed as dist
import time

INITIAL_LOG_LOSS_SCALE = 20.0
torch.autograd.set_detect_anomaly(True)

class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False
        self.fp16_scale_growth = 1e-3
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * (dist.get_world_size() if dist.is_initialized() else 1) * self.gradient_accumulation_steps
        self.num_steps = args.num_steps
        self.accum_iter = 0
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()
        self.save_dir = args.save_dir
        self.overwrite = args.overwrite
        self._load_and_sync_parameters()
        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.model.to(self.device)

        self.use_ddp = dist.is_initialized() and dist.get_world_size() > 1

        if self.use_ddp:
            print(f"Using DDP on {dist.get_world_size()} GPUs.")
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist.get_rank()],
                output_device=dist.get_rank(),
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        else:
            self.ddp_model = self.model

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.ddp_model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )


        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt,
                                                step_size = 10000,
                                                gamma = 0.99)

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.


        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)

        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

    def _load_and_sync_parameters(self):
        self.resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")

            state_dict = dist_util.load_state_dict(
                self.resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict['model_avg']
                state_dict = self._remove_module_prefix(state_dict)
                load_model(self.model, state_dict)
                if hasattr(self, 'model_avg'):
                    state_dict_avg = self._remove_module_prefix(state_dict_avg)
                    load_model(self.model_avg, state_dict_avg)
            else:
                state_dict = self._remove_module_prefix(state_dict)
                load_model(self.model, state_dict)
                if self.args.use_ema and hasattr(self, 'model_avg'):
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict())

    def _remove_module_prefix(self, state_dict):
        """Remove the prefix added by DistributedDataParallel."""
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def _load_optimizer_state(self):
        opt_checkpoint = self.find_resume_opt_checkpoint()
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    self.scaler.load_state_dict(state_dict['scaler'])
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd

    def run_loop(self):
        is_main = not dist.is_initialized() or dist.get_rank() == 0

        if is_main:
            print('train steps:', self.num_steps)

        epoch = 0
        while self.total_step() < self.num_steps:
            if is_main:
                print(f'Starting a new epoch {epoch} at step {self.total_step()}')

            if hasattr(self.data, 'sampler') and hasattr(self.data.sampler, 'set_epoch'):
                self.data.sampler.set_epoch(epoch)

            if is_main:
                data_iter = tqdm(self.data)
            else:
                data_iter = self.data

            for motion, cond in data_iter:
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val
                            for key, val in cond['y'].items()}

                self.run_step(motion, cond)

                if self.total_step() % self.log_interval == 0 and is_main:
                    for k, v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))
                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k, value=v, iteration=self.total_step(), group_name='Loss')

                if (self.total_step() % self.save_interval == 0 and self.total_step() != 0) or \
                (self.total_step() == self.num_steps - 1):
                    if dist.is_initialized():
                        dist.barrier()

                    self.save()
                    if is_main:
                        self.model.eval()
                        self.evaluate()
                        self.generate_during_training()
                        self.model.train()

                    if dist.is_initialized():
                        dist.barrier()

                self.step += 1

                if self.total_step() == self.num_steps:
                    break

            epoch += 1

    def generate_during_training(self):
        if not self.args.gen_during_training:
            return
        gen_args = copy.deepcopy(self.args)
        gen_args.model_path = os.path.join(self.save_dir, self.ckpt_file_name())
        gen_args.output_dir = os.path.join(self.save_dir, f'{self.ckpt_file_name()}.samples')
        gen_args.num_samples = self.args.gen_num_samples
        gen_args.num_repetitions = self.args.gen_num_repetitions
        gen_args.motion_length = 6.0
        gen_args.load_from_model_name = True
        all_objects = self.data.dataset.motion_dataset.cond_dict.keys()
        random.seed(self.step)
        gen_args.object_type = random.sample(all_objects, gen_args.num_samples)
        random.seed(self.args.seed)
        all_sample_save_path = generate(gen_args, self.data.dataset.motion_dataset.cond_dict)
        self.train_platform.report_media(title='Motion', series='Predicted Motion', iteration=self.total_step(),
                                         local_path=all_sample_save_path)


    def total_step(self):
        total_step = self.step
        if self.resume_step:
            # we add 1 because self.resume_step has already been done and we don't want to run it again
            # in particular we don't want to run the evaluation and generation again
            total_step += self.resume_step + 1
        return total_step

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        print(f'Evaluation during training no implemented')


    def run_step(self, batch, cond, epoch=-1):
        is_accumulating = (self.accum_iter + 1) % self.gradient_accumulation_steps != 0

        self.forward_backward(batch, cond, epoch, is_accumulating)

        self.accum_iter += 1

        if not is_accumulating:
            self.mp_trainer.optimize(self.opt, self.lr_scheduler)
            self._anneal_lr()
            self.log_step()
            self.accum_iter = 0

    def forward_backward(self, batch, cond, epoch, is_accumulating=False):
        if self.accum_iter == 0:
            self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond
            )

            if self.use_ddp and (is_accumulating or not last_batch):
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            else:
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            loss = loss / self.gradient_accumulation_steps

            if not is_accumulating:
                log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )
            self.mp_trainer.backward(loss)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
                for e in clip_weights:
                    del state_dict[e]

            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = {}
                for i, (name, _) in enumerate(self.model.named_parameters()):
                    state_dict[name] = self.mp_trainer.master_params[i]

            del_clip(state_dict)

            if self.args.use_ema and hasattr(self, 'model_avg'):
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.log(f"saving model...")
                filename = self.ckpt_file_name()
                with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint()

        if not dist.is_initialized() or dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.save_dir, f"opt{(self.total_step()):09d}.pt"),
                "wb",
            ) as f:
                opt_state = self.opt.state_dict()
                if self.use_fp16:
                    opt_state = {
                        'opt': opt_state,
                        'scaler': self.scaler.state_dict(),
                    }
                torch.save(opt_state, f)

    def find_resume_checkpoint(self) -> Optional[str]:
        """Return the model checkpoint with the highest step number."""

        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None

    def find_resume_opt_checkpoint(self) -> Optional[str]:
        """Return the optimizer checkpoint with the highest step number."""

        matches = {file: re.match(r'opt(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None




def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return logger.get_dir()


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
