import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import Truebones

def get_dataset_class(name):
    return Truebones

def get_dataset(num_frames, split='train', temporal_window=31, balanced=False,
                objects_subset="all", data_dir=""):
    dataset = Truebones(
        split=split,
        num_frames=num_frames,
        temporal_window=temporal_window,
        balanced=balanced,
        objects_subset=objects_subset,
        data_dir=data_dir,
    )
    return dataset


def get_dataset_loader(batch_size, num_frames, split='train', temporal_window=31,
                       balanced=True, objects_subset="all", data_dir=""):
    dataset = get_dataset(
        num_frames=num_frames,
        split=split,
        temporal_window=temporal_window,
        balanced=balanced,
        objects_subset=objects_subset,
        data_dir=data_dir,
    )
    collate = truebones_batch_collate

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=42,
            drop_last=True,
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
