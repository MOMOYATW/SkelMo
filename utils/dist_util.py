"""
Helpers for distributed training.
"""

import socket
import os
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

used_device = 0

def setup_dist(device=0):
    """
    Setup a distributed process group.
    """
    global used_device
    used_device = device
    if dist.is_initialized():
        return
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        backend = "nccl" if th.cuda.is_available() else "gloo"

        print(f"[torchrun] Initializing rank {rank}/{world_size}, local_rank {local_rank}, backend {backend}")

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank
        )

        if th.cuda.is_available():
            th.cuda.set_device(local_rank)
            used_device = local_rank

        print(f"[torchrun] Rank {rank} initialized successfully")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    global used_device
    if th.cuda.is_available() and used_device>=0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
