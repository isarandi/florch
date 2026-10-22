"""Determinism and resume of the streams."""

import itertools
import os
import subprocess
import sys

import pytest

from florch.streams import DataStream, MultiStream, RoundRobinStream


def load_fn(item, rng=None, group_rng=None):
    return item


def item_ids(stream_iter, n):
    return [args[0] for _, args, _ in itertools.islice(stream_iter, n)]


def make_streams():
    data = DataStream([f'd{i}' for i in range(37)], load_fn, seed=3)
    roundrobin = RoundRobinStream(
        sections=[[f'a{i}' for i in range(20)], [f'b{i}' for i in range(15)]],
        sizes=[5, 3], load_fn=load_fn, seed=4, micro_batch_size=4, grad_accum_steps=2,
    )
    grouped = DataStream(
        [[f'g{i}_{v}' for v in range(4)] for i in range(10)], load_fn, seed=5,
        group_sample_size=2,
    )
    multi = MultiStream(
        [DataStream([f'p{i}' for i in range(11)], load_fn), roundrobin, grouped],
        micro_batch_sizes=[3, 4, 2], seed=6,
    )
    return {'data': data, 'roundrobin': roundrobin, 'grouped': grouped, 'multi': multi}


@pytest.mark.parametrize('name', ['data', 'roundrobin', 'grouped', 'multi'])
def test_seek_equals_islice(name):
    stream = make_streams()[name]
    reference = item_ids(iter(stream), 300)
    for skip in (0, 1, 7, 36, 37, 100, 211):
        assert item_ids(stream.seek(skip), 50) == reference[skip:skip + 50], skip


SUBPROCESS_SCRIPT = """
import itertools, json, sys
sys.path.insert(0, {tests_dir!r})
from test_streams_and_loop import make_streams, item_ids
print(json.dumps({{k: item_ids(s.seek(13), 40) for k, s in make_streams().items()}}))
"""


def test_order_is_identical_across_processes():
    """String hashing is randomized per process; derived seeds must not depend on it."""
    script = SUBPROCESS_SCRIPT.format(tests_dir=os.path.dirname(__file__))
    outputs = []
    for hashseed in ('1', '2'):
        env = {**os.environ, 'PYTHONHASHSEED': hashseed}
        result = subprocess.run(
            [sys.executable, '-c', script], env=env, capture_output=True, text=True,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
