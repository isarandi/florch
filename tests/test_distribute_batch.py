"""Tests for distribute_batch.

The central property is that no data section may be dropped for a whole run.
The allocation is computed once and then cycled, so a section that gets zero
slots in the repeating unit is never sampled again -- silently, since nothing
about the training loop looks wrong.
"""

import itertools
import random

import pytest

from florch.util.distribute_batch import distribute_batch


def section_totals(flat, n_sections):
    """Per-section totals over one period of the allocation."""
    return [sum(flat[i::n_sections]) for i in range(n_sections)]


def micro_batch_sizes(flat, n_sections):
    """Item count of each micro-batch in the period."""
    return [
        sum(flat[i * n_sections : (i + 1) * n_sections]) for i in range(len(flat) // n_sections)
    ]


def random_configs(n, seed):
    rng = random.Random(seed)
    while n > 0:
        n_sections = rng.randint(2, 6)
        weights = [rng.randint(1, 30) for _ in range(n_sections)]
        accum = rng.randint(1, 10)
        micro = sum(weights) // accum
        if micro <= 0:
            continue
        n -= 1
        yield weights, micro, accum


# =============================================================================
# The bug: a section starved for the entire run
# =============================================================================


@pytest.mark.parametrize(
    'weights,accum', [([1, 1, 1], 2), ([14, 6, 2], 8), ([3, 18], 7), ([1, 1, 5], 3)]
)
def test_no_section_is_starved(weights, accum):
    """These configs used to allocate zero slots to one section, forever."""
    micro = sum(weights) // accum
    totals = section_totals(distribute_batch(weights, micro, accum), len(weights))
    assert all(t > 0 for t in totals), f'section starved: {totals}'


def test_no_section_is_starved_over_random_configs():
    for weights, micro, accum in random_configs(2000, seed=0):
        totals = section_totals(distribute_batch(weights, micro, accum), len(weights))
        assert all(t > 0 for t in totals), f'{weights} micro={micro} accum={accum} -> {totals}'


# =============================================================================
# Invariants the batcher and the resume arithmetic depend on
# =============================================================================


def test_every_micro_batch_has_exactly_micro_batch_size_items():
    """Batcher(drop_last=True) and the resume skip both assume a fixed size."""
    for weights, micro, accum in random_configs(1000, seed=1):
        flat = distribute_batch(weights, micro, accum)
        assert all(s == micro for s in micro_batch_sizes(flat, len(weights)))


def test_period_is_a_whole_number_of_accumulation_windows():
    for weights, micro, accum in random_configs(1000, seed=2):
        flat = distribute_batch(weights, micro, accum)
        n_micro_batches = len(flat) // len(weights)
        assert n_micro_batches % accum == 0


def test_long_run_rate_matches_the_weights():
    """Over one period, each section's share equals its weight exactly."""
    for weights, micro, accum in random_configs(1000, seed=3):
        totals = section_totals(distribute_batch(weights, micro, accum), len(weights))
        total_items = sum(totals)
        total_weight = sum(weights)
        for got, weight in zip(totals, weights):
            assert got / total_items == pytest.approx(weight / total_weight)


def test_allocation_is_a_pure_function_of_its_arguments():
    """Resume regenerates the schedule from scratch, so it must not drift."""
    for weights, micro, accum in random_configs(200, seed=4):
        assert distribute_batch(weights, micro, accum) == distribute_batch(weights, micro, accum)


# =============================================================================
# Configs that already worked must keep their exact schedule
# =============================================================================

# Captured from the implementation before the starvation fix, so that a change
# in data order for these configs shows up as a test failure rather than as an
# unexplained difference between two training runs.
LEGACY_SCHEDULES = [
    ([4, 4], 2, [2, 2, 2, 2]),
    ([10, 5], 3, [3, 2, 3, 2, 4, 1]),
    ([16, 12, 19, 7], 3, [5, 4, 6, 3, 5, 5, 6, 2, 6, 3, 7, 2]),
    ([4, 3, 11, 16], 2, [3, 1, 5, 8, 1, 2, 6, 8]),
    ([5, 18, 15], 2, [3, 9, 7, 2, 9, 8]),
]


@pytest.mark.parametrize('weights,accum,expected', LEGACY_SCHEDULES)
def test_exactly_representable_configs_keep_their_legacy_schedule(weights, accum, expected):
    assert sum(weights) % accum == 0, 'this test is about the exactly representable case'
    micro = sum(weights) // accum
    assert distribute_batch(weights, micro, accum) == expected


# =============================================================================
# Argument validation
# =============================================================================


@pytest.mark.parametrize(
    'weights,micro,accum',
    [([1, 1], 0, 4), ([1, 1], -1, 2), ([0, 0], 1, 1), ([1, -1], 1, 1), ([1, 1], 1, 0)],
)
def test_invalid_arguments_raise(weights, micro, accum):
    with pytest.raises(ValueError):
        distribute_batch(weights, micro, accum)


# =============================================================================
# End to end through RoundRobinStream
# =============================================================================


def test_stream_samples_every_section_and_resumes_identically():
    from florch.streams import RoundRobinStream

    def load_fn(item, rng=None, group_rng=None):
        return item

    sections = [[f'{name}{i}' for i in range(50)] for name in 'abc']
    stream = RoundRobinStream(
        sections=sections, sizes=[14, 6, 2], load_fn=load_fn, seed=7, grad_accum_steps=8
    )

    items = [args[0] for _, args, _ in itertools.islice(iter(stream), 400)]
    for name in 'abc':
        assert any(x.startswith(name) for x in items), f'section {name} never sampled'

    for skip in (0, 1, 37, 128, 251):
        tail = [args[0] for _, args, _ in itertools.islice(stream.seek(skip), 60)]
        assert tail == items[skip : skip + 60], f'resume at {skip} diverges'
