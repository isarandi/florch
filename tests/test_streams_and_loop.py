"""Determinism, resume and distributed splitting of the streams; norm loss settings."""

import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

from florch.streams import DataStream, MultiStream, RoundRobinStream, stream_to_loader


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


def test_ranks_partition_each_global_batch():
    """The ranks' micro-batches together form the global batch, and each rank gets
    items of every source."""
    world_size, micro = 2, 6
    stream = MultiStream(
        [DataStream([f'p{i}' for i in range(50)], load_fn, seed=1),
         DataStream([f'k{i}' for i in range(50)], load_fn, seed=2)],
        micro_batch_sizes=[8, 4], seed=0,
    )
    global_items = item_ids(iter(stream), 5 * world_size * micro)
    per_rank = []
    for rank in range(world_size):
        loader_input = _loader_items(stream, micro, rank, world_size, n_batches=5)
        per_rank.append(loader_input)

    for b in range(5):
        global_batch = global_items[b * world_size * micro:(b + 1) * world_size * micro]
        rank_batches = [r[b * micro:(b + 1) * micro] for r in per_rank]
        assert sorted(sum(rank_batches, [])) == sorted(global_batch)
        for batch in rank_batches:
            assert any(x.startswith('p') for x in batch)
            assert any(x.startswith('k') for x in batch)


def make_grouped_multistream(group_size=4):
    """A single-view stream next to a multi-view stream whose items are views of frames.

    Item names are '<frame>/<view>', so the frame of each item can be read back.
    """
    frames = [[f'f{i}/v{v}' for v in range(6)] for i in range(40)]
    multiview = RoundRobinStream(
        sections=[frames[:25], frames[25:]],
        sizes=[2 * group_size, group_size], load_fn=load_fn, seed=7,
        group_sample_size=group_size,
    )
    single = DataStream([f's{i}' for i in range(30)], load_fn, seed=8)
    return MultiStream([single, multiview], micro_batch_sizes=[8, 16], seed=9)


@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_contiguous_rank_split_keeps_groups_whole(world_size):
    group_size, n_batches = 4, 6
    stream = make_grouped_multistream(group_size)
    full = 8 + 16
    micro = full // world_size
    global_items = item_ids(iter(stream), n_batches * full)
    per_rank = [
        _loader_items(stream, micro, rank, world_size, n_batches)
        for rank in range(world_size)
    ]

    for b in range(n_batches):
        global_batch = global_items[b * full:(b + 1) * full]
        single_block, mv_block = global_batch[:8], global_batch[8:]
        rank_batches = [r[b * micro:(b + 1) * micro] for r in per_rank]
        for rank, batch in enumerate(rank_batches):
            # Contiguous pieces of each stream's block, in stream order.
            s_share, mv_share = 8 // world_size, 16 // world_size
            assert batch[:s_share] == single_block[rank * s_share:(rank + 1) * s_share]
            assert batch[s_share:] == mv_block[rank * mv_share:(rank + 1) * mv_share]
            mv_items = batch[s_share:]
            for g in range(0, len(mv_items), group_size):
                frames = {x.split('/')[0] for x in mv_items[g:g + group_size]}
                assert len(frames) == 1, (rank, mv_items)
        assert sorted(sum(rank_batches, [])) == sorted(global_batch)


def test_world_size_one_is_unchanged():
    stream = make_grouped_multistream()
    assert _loader_items(stream, 24, 0, 1, 5) == item_ids(iter(stream), 5 * 24)


def test_rank_split_resumes_like_a_fresh_run():
    stream = make_grouped_multistream()
    world_size, micro = 2, 12
    full_run = _loader_items(stream, micro, 1, world_size, n_batches=8)
    resumed = _loader_items(
        stream, micro, 1, world_size, n_batches=5, completed_steps=3
    )
    assert resumed == full_run[3 * micro:]


def test_block_not_divisible_by_world_size_raises():
    stream = MultiStream(
        [DataStream([f'p{i}' for i in range(50)], load_fn, seed=1),
         DataStream([f'k{i}' for i in range(50)], load_fn, seed=2)],
        micro_batch_sizes=[7, 5], seed=0,
    )
    with pytest.raises(ValueError, match='grad_accum_steps \\* world_size'):
        _loader_items(stream, 6, 0, 2, n_batches=1)


def test_rank_share_not_whole_groups_raises():
    # 16 multi-view items per micro-batch on 8 ranks: 2 per rank, half a group.
    stream = make_grouped_multistream(group_size=4)
    with pytest.raises(ValueError, match='group size 4'):
        _loader_items(stream, 3, 0, 8, n_batches=1)
    # The same rule applies to a plain grouped stream, as used by the test loader.
    grouped = DataStream(
        [[f'g{i}_{v}' for v in range(4)] for i in range(10)], load_fn, seed=5,
        group_sample_size=4,
    )
    with pytest.raises(ValueError, match='group size 4'):
        _loader_items(grouped, 2, 0, 2, n_batches=1)


def test_block_sizes_must_match_micro_batch():
    stream = make_grouped_multistream()
    with pytest.raises(ValueError, match='sum to 24'):
        _loader_items(stream, 10, 0, 2, n_batches=1)


def test_roundrobin_sizes_must_be_whole_groups():
    stream = RoundRobinStream(
        sections=[[[f'a{i}_{v}' for v in range(4)] for i in range(5)]],
        sizes=[6], load_fn=load_fn, group_sample_size=4,
    )
    with pytest.raises(ValueError, match='multiples of group_sample_size'):
        next(iter(stream))


def _loader_items(stream, micro, rank, world_size, n_batches, completed_steps=0):
    """The items stream_to_loader would feed to the batcher on this rank."""
    import florch.data_loading

    captured = []

    def fake_loader(stream_iter, batch_size, n_workers=None):
        assert batch_size == micro
        captured.extend(item_ids(stream_iter, n_batches * micro))

    original = florch.data_loading.function_calls_to_batched_torch_loader
    florch.data_loading.function_calls_to_batched_torch_loader = fake_loader
    try:
        stream_to_loader(
            stream, micro_batch_size=micro, rank=rank, world_size=world_size,
            completed_steps=completed_steps,
        )
    finally:
        florch.data_loading.function_calls_to_batched_torch_loader = original
    return captured


def test_norm_loss_hooks_cover_batch_renorm():
    from florch.layers import BatchRenorm2d
    from florch.training.loop import NORM_LAYER_TYPES

    assert isinstance(BatchRenorm2d(4), NORM_LAYER_TYPES)


def test_norm_loss_ramp_is_a_length():
    from florch.util.misc import smootherstep

    start, length = 1000, 4000
    assert smootherstep(start, start, length=length) == 0
    assert smootherstep(start + length / 2, start, length=length) == pytest.approx(0.5)
    assert smootherstep(start + length, start, length=length) == 1
    assert smootherstep(start - 1, start, length=length) == 0
    assert np.isclose(smootherstep(start + length + 1, start, length=length), 1)


def test_grad_norm_keys_use_group_names():
    from florch.training.loop import _named_clip_groups

    backbone, head = [object()], [object()]
    assert _named_clip_groups({'backbone': backbone, 'head': head}) == (
        ['backbone', 'head'], [backbone, head]
    )
    assert _named_clip_groups([backbone, head]) == ([0, 1], [backbone, head])


def test_checkpoint_without_clipper_state_starts_histories_at_0_7(tmp_path):
    import torch

    from florch import TrainingConfig
    from florch.grad_clipping import AdaptiveGradientClipper
    from florch.training.job import TrainingJob

    class Job(TrainingJob):
        def build_data(self):
            raise NotImplementedError

        def build_model(self):
            model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
            model.backbone, model.head = model[0], model[1]
            return model

        def build_trainer(self, model):
            raise NotImplementedError

        def build_optimizer(self):
            raise NotImplementedError

        def get_grad_clip_param_groups(self):
            return {
                'backbone': self.model.backbone.parameters(),
                'head': self.model.head.parameters(),
            }

    ckpt_path = tmp_path / 'weights.pth'
    torch.save({'model_state_dict': Job.build_model(None).state_dict()}, ckpt_path)
    config = TrainingConfig(
        training_steps=10, init_path=str(ckpt_path), clip_grad_norm_factor=2.0,
        clip_grad_norm_histsize=5, multi_gpu=False,
    )
    job = Job(config)
    job.restore()
    assert isinstance(job.grad_clipper, AdaptiveGradientClipper)
    assert [list(h) for h in job.grad_clipper.histories] == [[0.7] * 5, [0.7] * 5]

    # A checkpoint with clipper state keeps its own histories.
    torch.save({
        'model_state_dict': job.model.state_dict(),
        'grad_clipper_state_dict': {'histories': [[1.5], [2.5]]},
    }, ckpt_path)
    job = Job(config)
    job.restore()
    assert [list(h) for h in job.grad_clipper.histories] == [[1.5], [2.5]]
