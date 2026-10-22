"""Seekable stream classes for deterministic, O(1)-resumable data loading.

Three levels of composition:
- DataStream: single dataset with shuffle + per-item RNG
- RoundRobinStream: weighted interleaving of sections (same load_fn)
- MultiStream: merge multiple streams (different load_fns)

All support seek(skip) for O(1) resume — no replaying of skipped items.

Example:
    stream = DataStream(image_paths, load_image, seed=42)
    loader = stream_to_loader(stream, micro_batch_size=64, completed_steps=1000)

    # Multi-dataset with round-robin mixing:
    stream = RoundRobinStream(
        sections=[coco_paths, h36m_paths, mpii_paths],
        sizes=[10, 5, 3],
        load_fn=load_pose,
        seed=42,
    )

    # Multi-task with different load functions:
    multi = MultiStream(
        streams=[stream_param, stream_kp, stream_dense],
        micro_batch_sizes=[16, 32, 8],
        seed=42,
    )
    loader = stream_to_loader(multi, micro_batch_size=56, completed_steps=5000, total_steps=100000)
"""

import itertools
import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


class DataStream:
    """Deterministic stream over items with per-item RNG.

    Iterates over items repeatedly (infinite stream), shuffling each epoch.
    Each item receives an independent numpy RNG for reproducible augmentation.

    Supports O(1) resume via seek(skip): uses position-based RNG seeding
    and epoch-based shuffling instead of sequential RNG chains.

    Args:
        items: Sequence of items (paths, indices, example objects, etc.)
            If group_sample_size is set, items should be groups (lists of items).
        load_fn: Function to load each item. Signature:
            load_fn(item, *extra_args, rng=rng, group_rng=group_rng, **extra_kwargs) -> dict
        seed: Random seed for deterministic ordering and RNG derivation.
        shuffle: Whether to shuffle items each epoch.
        extra_args: Additional positional args passed to load_fn.
        extra_kwargs: Additional keyword args passed to load_fn.
        group_sample_size: If set, items are treated as groups and this many
            samples are drawn from each group per pass. All samples from a group
            share the same group_rng (useful for synchronized augmentation).
    """

    def __init__(
        self,
        items: Sequence[Any],
        load_fn: Callable,
        seed: int = 0,
        shuffle: bool = True,
        extra_args: Tuple = (),
        extra_kwargs: Optional[Dict] = None,
        group_sample_size: Optional[int] = None,
    ):
        self.items = list(items)
        self.load_fn = load_fn
        self.seed = seed
        self.shuffle = shuffle
        self.extra_args = extra_args
        self.extra_kwargs = extra_kwargs or {}
        self.group_sample_size = group_sample_size

    def __iter__(self) -> Iterator[Tuple[Callable, Tuple, Dict]]:
        yield from self.seek(0)

    def seek(self, skip: int = 0) -> Iterator[Tuple[Callable, Tuple, Dict]]:
        """Yield (load_fn, args, kwargs) tuples from position `skip`. O(1) startup."""
        if self.group_sample_size is not None:
            yield from self._seek_grouped(skip)
        else:
            yield from self._seek_simple(skip)

    def _seek_simple(self, skip: int):
        N = len(self.items)
        epoch_start = skip // N
        pos_start = skip % N
        seed = self.seed

        for epoch in itertools.count(epoch_start):
            if self.shuffle:
                order = np.random.default_rng(
                    _derive_seed(seed, 'shuffle', epoch)
                ).permutation(N)
                items = [self.items[i] for i in order]
            else:
                items = self.items

            start = pos_start if epoch == epoch_start else 0
            for local_idx in range(start, N):
                global_idx = epoch * N + local_idx
                yield (
                    self.load_fn,
                    (items[local_idx], *self.extra_args),
                    {
                        **self.extra_kwargs,
                        'rng': np.random.default_rng(
                            _derive_seed(seed, 'item', global_idx)
                        ),
                        'group_rng': np.random.default_rng(
                            _derive_seed(seed, 'group', global_idx)
                        ),
                    },
                )

    def _seek_grouped(self, skip: int):
        N = len(self.items)
        S = self.group_sample_size
        items_per_epoch = N * S
        epoch_start = skip // items_per_epoch
        pos_start = skip % items_per_epoch
        seed = self.seed

        for epoch in itertools.count(epoch_start):
            if self.shuffle:
                order = np.random.default_rng(
                    _derive_seed(seed, 'shuffle', epoch)
                ).permutation(N)
                groups = [self.items[i] for i in order]
            else:
                groups = self.items

            start_group = pos_start // S if epoch == epoch_start else 0
            start_sample = pos_start % S if epoch == epoch_start else 0

            for group_idx in range(start_group, N):
                group = groups[group_idx]
                indices = np.random.default_rng(
                    _derive_seed(seed, 'sample', epoch, group_idx)
                ).choice(len(group), size=S, replace=False)
                group_rng = np.random.default_rng(
                    _derive_seed(seed, 'group', epoch, group_idx)
                )

                sample_start = start_sample if group_idx == start_group and epoch == epoch_start else 0
                for sample_idx in range(sample_start, S):
                    global_idx = epoch * items_per_epoch + group_idx * S + sample_idx
                    yield (
                        self.load_fn,
                        (group[indices[sample_idx]], *self.extra_args),
                        {
                            **self.extra_kwargs,
                            'rng': np.random.default_rng(
                                _derive_seed(seed, 'item', global_idx)
                            ),
                            'group_rng': group_rng,
                        },
                    )

            # Reset for subsequent epochs
            pos_start = 0

    def __repr__(self):
        return (
            f"DataStream({len(self.items)} items, "
            f"load_fn={self.load_fn.__name__}, seed={self.seed})"
        )


class RoundRobinStream:
    """Weighted round-robin over multiple data sections with the same load function.

    Each section is a list of items. Items are drawn from sections in round-robin
    fashion according to the given sizes. Supports O(1) seek via position decomposition.

    Args:
        sections: List of item lists, one per section.
        sizes: List of ints, how many items to draw from each section per round.
        load_fn: Loading function (same for all sections).
        seed: Random seed.
        shuffle: Whether to shuffle within each section.
        extra_args: Additional args for load_fn.
        extra_kwargs: Additional kwargs for load_fn.
        micro_batch_size: How many items this stream contributes to each micro-batch.
            If given, the sizes are treated as relative weights and redistributed so
            that every micro-batch receives (nearly) the same mix of sections. If None,
            the raw sizes are cycled as they are, which keeps each section's count
            intact (e.g. whole multi-view groups when sizes are multiples of
            group_sample_size).
        grad_accum_steps: Gradient accumulation steps; only used together with
            micro_batch_size.
        group_sample_size: Passed to sub-DataStreams.
    """

    def __init__(
        self,
        sections: List[Sequence[Any]],
        sizes: List[int],
        load_fn: Callable,
        seed: int = 0,
        shuffle: bool = True,
        extra_args: Tuple = (),
        extra_kwargs: Optional[Dict] = None,
        micro_batch_size: Optional[int] = None,
        grad_accum_steps: int = 1,
        group_sample_size: Optional[int] = None,
    ):
        self.sections = sections
        self.raw_sizes = list(sizes)
        self.load_fn = load_fn
        self.seed = seed
        self.shuffle = shuffle
        self.extra_args = extra_args
        self.extra_kwargs = extra_kwargs or {}
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.group_sample_size = group_sample_size

    def _effective_sizes(self) -> List[int]:
        if self.micro_batch_size is None:
            return self.raw_sizes
        from florch.util.distribute_batch import distribute_batch
        # Allocating over 4 accumulation windows (not 1) lets each micro-batch get
        # its share even when a section's share per window is fractional. The
        # factor is part of the schedule: changing it changes the data order.
        return distribute_batch(
            self.raw_sizes, self.micro_batch_size, 4 * self.grad_accum_steps
        )

    def _make_sub_streams(self):
        return [
            DataStream(
                items=section,
                load_fn=self.load_fn,
                seed=_derive_seed(self.seed, 'section', i),
                shuffle=self.shuffle,
                extra_args=self.extra_args,
                extra_kwargs=self.extra_kwargs,
                group_sample_size=self.group_sample_size,
            )
            for i, section in enumerate(self.sections)
        ]

    def __iter__(self) -> Iterator[Tuple[Callable, Tuple, Dict]]:
        yield from self.seek(0)

    def seek(self, skip: int = 0) -> Iterator[Tuple[Callable, Tuple, Dict]]:
        """Seek all sub-streams and resume roundrobin from position."""
        effective_sizes = self._effective_sizes()
        n_streams = len(self.sections)
        sub_streams = self._make_sub_streams()

        positions, start_seg, start_off = _roundrobin_decompose(
            effective_sizes, n_streams, skip
        )

        iterators = [s.seek(positions[i]) for i, s in enumerate(sub_streams)]
        yield from _roundrobin_seeked(iterators, effective_sizes, n_streams, start_seg, start_off)

    def __repr__(self):
        sections_str = ', '.join(f'{len(s)}' for s in self.sections)
        return (
            f"RoundRobinStream([{sections_str}] items, "
            f"sizes={self.raw_sizes}, seed={self.seed})"
        )


class MultiStream:
    """Merge multiple streams via round-robin with given micro-batch sizes.

    Each stream can have a different load function. Child stream seeds are
    derived deterministically from this stream's seed.

    Args:
        streams: List of DataStream, RoundRobinStream, or other seekable streams.
        micro_batch_sizes: How many items each stream contributes per substep.
        seed: Seed for deriving child stream seeds.
    """

    def __init__(
        self,
        streams: List,
        micro_batch_sizes: List[int],
        seed: int = 0,
    ):
        self.streams = streams
        self.micro_batch_sizes = micro_batch_sizes
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[Callable, Tuple, Dict]]:
        yield from self.seek(0)

    def seek(self, skip: int = 0) -> Iterator[Tuple[Callable, Tuple, Dict]]:
        """Decompose skip across child streams, seek each, roundrobin from position."""
        n_streams = len(self.streams)

        # Override child seeds deterministically
        seeded_streams = []
        for i, stream in enumerate(self.streams):
            derived_seed = _derive_seed(self.seed, 'stream', i)
            seeded_streams.append(_with_seed(stream, derived_seed))

        positions, start_seg, start_off = _roundrobin_decompose(
            self.micro_batch_sizes, n_streams, skip
        )

        iterators = [s.seek(positions[i]) for i, s in enumerate(seeded_streams)]
        yield from _roundrobin_seeked(
            iterators, self.micro_batch_sizes, n_streams, start_seg, start_off
        )

    def __repr__(self):
        total = sum(self.micro_batch_sizes)
        return (
            f"MultiStream({len(self.streams)} streams, "
            f"micro_batch_sizes={self.micro_batch_sizes}, total={total})"
        )


# === Loader ===


def stream_to_loader(
    stream,
    micro_batch_size: int,
    num_workers: Optional[int] = None,
    completed_steps: int = 0,
    total_steps: Optional[int] = None,
    grad_accum_steps: int = 1,
    rank: int = 0,
    world_size: int = 1,
):
    """Convert a stream to a parallel batched torch loader.

    Uses seek() for O(1) resume if the stream supports it,
    otherwise falls back to islice (O(n_skip)).

    Args:
        stream: DataStream, RoundRobinStream, MultiStream, or any iterable
            of (load_fn, args, kwargs) tuples.
        micro_batch_size: Items per micro-batch (per substep, per rank).
        num_workers: Parallel workers for data loading. Default: min(cpu_count, 12).
        completed_steps: Steps already completed (for resume).
        total_steps: Total training steps. Limits the stream length.
        grad_accum_steps: Gradient accumulation steps.
        rank: Process rank for distributed training.
        world_size: Number of processes.

    Returns:
        torchdata.nodes.Loader yielding batched, collated tensors.
    """
    from florch.data_loading import function_calls_to_batched_torch_loader

    full_batch_size = micro_batch_size * world_size
    n_skip = completed_steps * grad_accum_steps * full_batch_size
    n_total = total_steps * grad_accum_steps * full_batch_size if total_steps else None
    n_remaining = n_total - n_skip if n_total is not None else None

    # Seek or fallback to islice
    if hasattr(stream, 'seek'):
        stream_iter = stream.seek(n_skip)
    else:
        stream_iter = itertools.islice(iter(stream), n_skip, n_total)
        n_remaining = None  # islice already limited

    # Limit total items
    if n_remaining is not None:
        stream_iter = itertools.islice(stream_iter, n_remaining)

    # Distributed: take contiguous chunks to preserve mixing ratios
    if world_size > 1:
        stream_iter = _batch_chunk(stream_iter, full_batch_size, rank, micro_batch_size)

    return function_calls_to_batched_torch_loader(
        stream_iter, micro_batch_size, n_workers=num_workers,
    )


# === Internal helpers ===


def _batch_chunk(stream_iter, full_batch_size, rank, per_rank_batch_size):
    """Yield contiguous chunks of each batch for distributed training.

    Preserves mixing ratios by giving each rank a contiguous slice of the
    full batch rather than interleaving individual items.
    """
    offset = rank * per_rank_batch_size
    while True:
        batch = list(itertools.islice(stream_iter, full_batch_size))
        if len(batch) < full_batch_size:
            break
        yield from batch[offset:offset + per_rank_batch_size]


_SEED_TAGS = {'shuffle': 0, 'item': 1, 'group': 2, 'sample': 3, 'section': 4, 'stream': 5}


def _derive_seed(parent_seed: int, tag: str, *indices: int) -> int:
    """Derive a deterministic child seed from a parent seed, a tag and indices.

    Uses SeedSequence spawn keys rather than Python's hash(), because string hashes
    are randomized per process (PYTHONHASHSEED), which would make shuffling differ
    between ranks and between a run and its resume. The spawn key keeps the tag and
    indices as separate words, so different (tag, indices) never collide by padding.
    """
    ss = np.random.SeedSequence(parent_seed, spawn_key=(_SEED_TAGS[tag], *indices))
    return int(ss.generate_state(1, np.uint64)[0])


def _with_seed(stream, seed: int):
    """Return a shallow copy of the stream with a different seed."""
    import copy
    s = copy.copy(stream)
    s.seed = seed
    # For MultiStream, don't override children — they'll get their own derived seeds
    return s


def _roundrobin_decompose(sizes, n_iterables, skip):
    """Decompose a global position into per-iterable positions.

    Handles the general case where len(sizes) != n_iterables
    (e.g. from distribute_batch with grad_accum_steps > 1).
    The roundrobin pattern repeats after lcm(n_iterables, len(sizes)) segments.

    Args:
        sizes: List of segment sizes for the roundrobin.
        n_iterables: Number of iterables being round-robined.
        skip: Global position to decompose.

    Returns:
        positions: List of per-iterable positions (length n_iterables).
        start_segment: Index of the segment within one cycle to resume from.
        start_offset: Position within that segment to resume from.
    """
    n_sizes = len(sizes)
    cycle_len = _lcm(n_iterables, n_sizes)

    # Build one full cycle of (iterable_index, segment_size) pairs
    cycle_segments = [
        (i % n_iterables, sizes[i % n_sizes]) for i in range(cycle_len)
    ]
    cycle_items = sum(s for _, s in cycle_segments)

    # Per-iterable contribution in one full cycle
    items_per_cycle = [0] * n_iterables
    for iter_idx, size in cycle_segments:
        items_per_cycle[iter_idx] += size

    # Complete cycles
    complete_cycles = skip // cycle_items if cycle_items > 0 else 0
    pos_in_cycle = skip % cycle_items if cycle_items > 0 else 0

    positions = [complete_cycles * c for c in items_per_cycle]

    # Walk through segments in the partial cycle
    remaining = pos_in_cycle
    for seg_idx, (iter_idx, size) in enumerate(cycle_segments):
        if remaining < size:
            positions[iter_idx] += remaining
            start_segment = seg_idx
            start_offset = remaining
            break
        remaining -= size
        positions[iter_idx] += size
    else:
        # Exhausted the cycle (remaining == 0 at cycle boundary)
        start_segment = 0
        start_offset = 0

    return positions, start_segment, start_offset


def _roundrobin_seeked(iterators, sizes, n_iterables, start_segment, start_offset):
    """Resume roundrobin from a specific point in the cycle.

    Args:
        iterators: List of iterators (one per iterable), already seeked.
        sizes: Segment sizes for roundrobin.
        n_iterables: Number of iterables.
        start_segment: Segment index to start from within the cycle.
        start_offset: Position within the start segment.
    """
    n_sizes = len(sizes)
    cycle_len = _lcm(n_iterables, n_sizes)

    def segment_iter(seg_idx):
        """Iterator that yields from the right iterable for a given segment."""
        iter_idx = seg_idx % n_iterables
        size = sizes[seg_idx % n_sizes]
        return iterators[iter_idx], size

    # First partial cycle: start from start_segment with start_offset
    for seg_idx in range(start_segment, cycle_len):
        iterator, size = segment_iter(seg_idx)
        start = start_offset if seg_idx == start_segment else 0
        for _ in range(size - start):
            yield next(iterator)

    # Full cycles thereafter
    while True:
        for seg_idx in range(cycle_len):
            iterator, size = segment_iter(seg_idx)
            for _ in range(size):
                yield next(iterator)


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)
