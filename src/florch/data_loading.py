"""Torchdata-based parallel data loading pipeline."""

import contextlib
import os

import numpy as np
import scipy.sparse
import torch
import torchdata.nodes


def function_calls_to_batched_torch_loader(fns_args_kwargs, batch_size, n_workers=None):
    n = torchdata.nodes.IterableWrapper(fns_args_kwargs)
    if n_workers is None:
        n_workers = min(len(os.sched_getaffinity(0)), 12)

    n = torchdata.nodes.ParallelMapper(
        n,
        map_fn=load_fn,
        in_order=True,
        num_workers=n_workers,
        method="process",
        multiprocessing_context="fork",
        snapshot_frequency=100,
    )
    n = torchdata.nodes.Batcher(n, batch_size=batch_size, drop_last=True)
    n = torchdata.nodes.ParallelMapper(
        n,
        map_fn=lambda x: merge(x, device='cpu'),
        num_workers=4,
        method="thread",
        in_order=True,
        snapshot_frequency=100,
    )
    n = torchdata.nodes.Prefetcher(n, 2, snapshot_frequency=100)
    n = torchdata.nodes.PinMemory(n, snapshot_frequency=100)
    return torchdata.nodes.Loader(n)


def load_fn(fn_args_kwargs):
    fn, args, kwargs = fn_args_kwargs
    return fn(*args, **kwargs)


def merge(batch, device=None):
    device = torch.device(device) if isinstance(device, str) else device
    if device is None:
        device = contextlib.nullcontext()

    with device:
        result = {}
        for element in batch:
            for key, value in element.items():
                append_nested(result, key, value)
        return stack_nested(key=None, value=result)


def append_nested(result_tree, key, value_tree):
    if isinstance(value_tree, dict):
        for subkey, subvalue in value_tree.items():
            append_nested(result_tree.setdefault(key, {}), subkey, subvalue)
    elif isinstance(value_tree, np.ndarray) or np.isscalar(value_tree):
        result_tree.setdefault(key, []).append(value_tree)
    elif isinstance(value_tree, scipy.sparse.csr_matrix):
        result_tree.setdefault(key, []).append(value_tree)
    else:
        raise RuntimeError(f"Unexpected type {type(value_tree)}")


def stack_nested(key, value):
    if isinstance(value, dict):
        return {
            subkey.removeprefix("_ragged_"): stack_nested(subkey, subvalue)
            for subkey, subvalue in value.items()
        }
    elif isinstance(value, list):
        if isinstance(value[0], str):
            return value
        elif isinstance(value[0], scipy.sparse.csr_matrix):
            return [scipy2torch_csr(v) for v in value]
        elif isinstance(value[0], (np.ndarray, np.number)):
            if key.startswith("_ragged_"):
                return torch.nested.nested_tensor(
                    [torch.as_tensor(v) for v in value])
            else:
                return torch.as_tensor(np.stack(value))
        elif isinstance(value[0], (int, float)):
            return torch.tensor(value)
        else:
            raise ValueError(f"Unsupported type: {type(value[0])}")
    else:
        raise RuntimeError(f"Unexpected type {type(value)}")


def scipy2torch_csr(csr):
    return torch.sparse_csr_tensor(
        torch.as_tensor(csr.indptr),
        torch.as_tensor(csr.indices),
        torch.as_tensor(csr.data),
        csr.shape,
    )
