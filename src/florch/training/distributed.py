"""Distributed training utilities."""

from typing import Union

import torch
import torch.distributed


def init_distributed() -> tuple:
    """Initialize distributed training.

    Returns:
        Tuple of (rank, world_size, device).
    """
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_node_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def get_world_size() -> int:
    """Get the number of processes in distributed training."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    """Get the rank of the current process."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def sync_metric(value: Union[float, torch.Tensor], world_size: int = None) -> float:
    """Synchronize a metric across all processes.

    Performs all_reduce with SUM and divides by world_size to get mean.

    Args:
        value: Scalar or tensor to synchronize.
        world_size: Number of processes. Auto-detected if None.

    Returns:
        Synchronized value as Python float.
    """
    if world_size is None:
        world_size = get_world_size()

    if torch.is_tensor(value):
        tensor = value.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            return tensor.item() / world_size
        else:
            return tensor.item()
    else:
        return float(value)
