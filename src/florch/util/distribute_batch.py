"""Distribute section sizes across micro-batches as evenly as possible.

Given section sizes (how many items from each data source per full batch),
a micro_batch_size (items per substep), and n_substeps (grad_accum_steps),
redistributes the items so each micro-batch has exactly micro_batch_size items
while preserving proportions as closely as possible.

When the sizes sum to exactly micro_batch_size * n_substeps, every section's
share is a whole number of items within a single accumulation window and the
original allocation below is used unchanged, so schedules stay bit-identical to
previous runs.

Otherwise the window cannot represent the proportions: the sum is truncated to
the window capacity, and a section whose share rounds away is then dropped for
the entire run, because the sizes list is computed once and cycled. For that
case the allocation instead repeats over several windows -- long enough that
every section's share over one period is a whole number of items -- and each
micro-batch is filled by giving each slot to whichever section is furthest
behind its ideal cumulative share. Micro-batches stay exactly micro_batch_size
items long, which the fixed-size batcher and the resume arithmetic both rely on.
"""

import math

import numpy as np


def distribute_batch(section_sizes, micro_batch_size, n_substeps):
    """Redistribute section sizes across micro-batches.

    Args:
        section_sizes: List of ints — how many items each section contributes
            per full batch (its weight; the sum need not divide evenly).
        micro_batch_size: Number of items per micro-batch (substep).
        n_substeps: Number of micro-batches (typically grad_accum_steps).

    Returns:
        Flat list of per-section counts, section-major within each micro-batch,
        covering one full period of the allocation. The caller cycles it.
    """
    weights = [int(w) for w in section_sizes]
    if any(w < 0 for w in weights):
        raise ValueError(f'Section sizes must be non-negative, got {section_sizes}')
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError('At least one section size must be positive')
    if micro_batch_size <= 0:
        raise ValueError(
            f'micro_batch_size must be positive, got {micro_batch_size}. With '
            f'section sizes summing to {total_weight} and {n_substeps} substeps '
            f'there is less than one item per micro-batch.'
        )
    if n_substeps <= 0:
        raise ValueError(f'n_substeps must be positive, got {n_substeps}')

    if micro_batch_size * n_substeps == total_weight:
        return _distribute_exact(weights, micro_batch_size, n_substeps)
    return _distribute_over_period(weights, micro_batch_size, n_substeps)


def _distribute_over_period(weights, micro_batch_size, n_substeps):
    """Allocate over as many windows as it takes for every share to come out whole."""
    total_weight = sum(weights)
    n_columns = _allocation_period(total_weight, micro_batch_size, n_substeps)

    # Column j gets micro_batch_size slots, each handed to the section furthest
    # below its ideal cumulative share. The ideal depends only on the column
    # index, so the schedule is reproducible from scratch and a resume at any
    # position regenerates the identical sequence.
    allocated = [0] * len(weights)
    columns = []
    for j in range(n_columns):
        ideal = [(j + 1) * micro_batch_size * w / total_weight for w in weights]
        counts = [0] * len(weights)
        for _ in range(micro_batch_size):
            k = max(
                range(len(weights)),
                key=lambda i: (ideal[i] - (allocated[i] + counts[i]), -i),
            )
            counts[k] += 1
        for k, c in enumerate(counts):
            allocated[k] += c
        columns.append(counts)

    return _shuffled_flat(columns)


def _allocation_period(total_weight, micro_batch_size, n_substeps):
    """Number of micro-batches after which the allocation pattern repeats.

    Short enough to stay cheap, long enough that every section's share over the
    period is a whole number of items — which is what stops a small section from
    rounding away to nothing. Rounded up to a multiple of n_substeps so the
    pattern stays aligned with accumulation windows.
    """
    columns = total_weight // math.gcd(total_weight, micro_batch_size)
    return math.lcm(columns, n_substeps)


def _distribute_exact(section_sizes, micro_batch_size, n_substeps):
    """The original allocation, for windows that can represent the proportions.

    Kept verbatim so that runs whose sizes sum to micro_batch_size * n_substeps
    -- which is every config this function handled correctly — keep producing
    byte-identical schedules, and only the previously broken configs change.
    """
    ds_parts = dict(enumerate(section_sizes))
    total_sum = sum(ds_parts.values())

    desired_totals = {
        k: round(n_substeps * micro_batch_size * v / total_sum) for k, v in ds_parts.items()
    }
    ok_totals = {
        k: np.ceil(n_substeps * micro_batch_size * v / total_sum) for k, v in ds_parts.items()
    }

    distributed = {}
    for k, total in desired_totals.items():
        avg, rem = divmod(total, n_substeps)
        distributed[k] = [avg] * n_substeps
        for i in range(rem):
            distributed[k][i] += 1

    def adjust_distribution(distr):
        for i in range(n_substeps):
            column = [distr[k][i] for k in distr]
            diff = micro_batch_size - sum(column)
            datasets_sorted = sorted(distr.keys(), key=lambda k: distr[k][i], reverse=(diff > 0))

            while True:
                for k in datasets_sorted:
                    if diff == 0:
                        break
                    if diff > 0 and sum(distr[k]) < ok_totals[k]:
                        distr[k][i] += 1
                        diff -= 1
                    elif diff < 0 and distr[k][i] > 0:
                        distr[k][i] -= 1
                        diff += 1
                if diff == 0:
                    break

    adjust_distribution(distributed)
    return _shuffled_flat([distributed[i] for i in range(len(section_sizes))], transpose=True)


def _shuffled_flat(columns, transpose=False):
    """Shuffle the micro-batch order within one period and flatten, section-major.

    The shuffle keeps successive accumulation windows from always seeing the
    sections in the same order; the fixed seed keeps it reproducible.
    """
    result = np.stack(columns)
    if transpose:
        result = result.T
    np.random.default_rng(42).shuffle(result, axis=0)
    return list(result.reshape(-1))
