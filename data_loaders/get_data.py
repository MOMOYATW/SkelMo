import math

import torch
import torch.distributed as dist
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones
from data_loaders.static_camera_dataset import StaticCameraDataset


class DistributedWeightedSampler(Sampler):
    """Deterministic weighted sampling whose global draw is sharded across ranks."""

    def __init__(self, weights, num_replicas, rank, seed=42):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

def get_dataset_class(name):
    return Truebones

def get_dataset(num_frames, split='train', temporal_window=31, balanced=False,
                objects_subset="all", data_dir="", **kwargs):
    if data_dir and (Path(data_dir) / "records").is_dir():
        return StaticCameraDataset(
            root=data_dir,
            split=split,
            num_frames=num_frames,
            temporal_window=temporal_window,
            balanced=balanced,
            objects_subset=objects_subset,
            **kwargs,
        )
    dataset = Truebones(
        split=split,
        num_frames=num_frames,
        temporal_window=temporal_window,
        balanced=balanced,
        objects_subset=objects_subset,
        data_dir=data_dir,
        **kwargs,
    )
    return dataset


def get_dataset_loader(batch_size, num_frames, split='train', temporal_window=31,
                       balanced=True, objects_subset="all", data_dir="", **kwargs):
    dataset = get_dataset(
        num_frames=num_frames,
        split=split,
        temporal_window=temporal_window,
        balanced=balanced,
        objects_subset=objects_subset,
        data_dir=data_dir,
        **kwargs,
    )
    collate = truebones_batch_collate

    sampling_weights = getattr(dataset, "sampling_weights", None) if balanced else None
    if dist.is_available() and dist.is_initialized():
        if sampling_weights is not None:
            sampler = DistributedWeightedSampler(
                sampling_weights,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                seed=42,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=42,
                drop_last=True,
            )
        shuffle = False
    elif sampling_weights is not None:
        sampler = WeightedRandomSampler(
            sampling_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=8,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
    )

    return loader
