import itertools
import os
from typing import Sequence

import numpy as np
import scipy.sparse
import simplepyutils as spu
import torch
import torchdata.nodes
import contextlib


MAX_INT = 2**32 - 1


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
    # n = torchdata.nodes.PinMemory(n, snapshot_frequency=100)
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


def collate_to_device(batch, device):
    grouped = spu.groupby_map(batch, lambda x: next(iter(x.items())))
    return {k: _collate_to_device(v, device) for k, v in grouped.items()}


def _collate_to_device(batch, device):
    out = {}
    device = torch.device(device) if isinstance(device, str) else device
    if device is None:
        device = contextlib.nullcontext()

    with device:
        if isinstance(batch[0], dict):
            for k in batch[0]:
                vs = [d[k] for d in batch]
                if isinstance(vs[0], str):
                    out[k] = vs
                elif isinstance(vs[0], (np.ndarray, np.number)):
                    if k.startswith("_ragged_"):
                        out[k] = torch.nested.nested_tensor(
                            [torch.as_tensor(v) for v in vs])
                    else:
                        out[k] = torch.as_tensor(np.stack(vs))
                elif isinstance(vs[0], scipy.sparse.csr_matrix):
                    out[k] = block_diag_csr([scipy2torch_csr(v) for v in vs])
                    # out[k] = scipy2torch_csr(scipy.sparse.block_diag(vs).tocsr())
                elif isinstance(vs[0], torch.Tensor):
                    if k.startswith("_ragged_"):
                        out[k] = torch.nested.nested_tensor(vs)
                    else:
                        out[k] = torch.stack(vs)
                elif isinstance(vs[0], (int, float)):
                    out[k] = torch.tensor(vs)
                elif isinstance(vs[0], (tuple, list, dict)):
                    out[k] = collate_to_device(vs, device)
                else:
                    raise ValueError(f"Unsupported type: {type(vs[0])}")
        elif isinstance(batch[0], tuple):
            out = tuple([collate_to_device(vs, device) for vs in zip(*batch)])
        elif isinstance(batch[0], list):
            out = [collate_to_device(vs, device) for vs in zip(*batch)]
        else:
            raise ValueError(f"Unsupported type: {type(batch[0])}")

    return out


def block_diag_csr(csr_mats: Sequence[torch.Tensor]):
    device = csr_mats[0].device
    crow_dtype = csr_mats[0].crow_indices().dtype
    col_dtype = csr_mats[0].col_indices().dtype

    row_offsets = torch.tensor(
        [0] + [csr.col_indices().shape[0] for csr in csr_mats], dtype=crow_dtype, device=device
    ).cumsum(0, dtype=crow_dtype)
    col_offsets = torch.tensor(
        [0] + [csr.shape[1] for csr in csr_mats], dtype=col_dtype, device=device
    ).cumsum(0, dtype=col_dtype)
    return torch.sparse_csr_tensor(
        torch.cat(
            [csr.crow_indices()[:-1] + offset for csr, offset in zip(csr_mats, row_offsets)]
            + [row_offsets[-1:]]
        ),
        torch.cat([csr.col_indices() + offset for csr, offset in zip(csr_mats, col_offsets)]),
        torch.cat([csr.values() for csr in csr_mats]),
        (sum(c.shape[0] for c in csr_mats), sum(c.shape[1] for c in csr_mats)),
    )


def scipy2torch_csr(csr):
    return torch.sparse_csr_tensor(
        torch.as_tensor(csr.indptr),
        torch.as_tensor(csr.indices),
        torch.as_tensor(csr.data),
        csr.shape,
    )


def new_rng(rng, advance_delta=0):
    gen = np.random.PCG64(rng.integers(0, MAX_INT))
    gen.advance(advance_delta)
    return np.random.Generator(gen)


def iterate_repeatedly(seq, shuffle_initially=True, shuffle_before_each_epoch=False, rng=None):
    """Iterates over and yields the elements of `iterable` over and over.
    If `shuffle_before_each_epoch` is True, the elements are put in a list and shuffled before
    every pass over the data, including the first."""

    if rng is None:
        rng = np.random.default_rng()

    # create a (shallow) copy so shuffling only applies to the copy.
    seq = list(seq)
    if shuffle_initially:
        rng.shuffle(seq)

    yield from seq

    while True:
        if shuffle_before_each_epoch:
            rng.shuffle(seq)
        yield from seq


def roundrobin(iterables, sizes):
    iterators = [iter(iterable) for iterable in iterables]
    for iterator, size in zip(itertools.cycle(iterators), itertools.cycle(sizes)):
        for _ in range(size):
            try:
                yield next(iterator)
            except StopIteration:
                return
