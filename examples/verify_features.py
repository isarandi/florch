"""Verify florch features: checkpointing, resume, gradient accumulation.

Usage:
    python examples/verify_features.py
    python examples/verify_features.py --ddp   # also run DDP smoke test (needs 2+ GPUs)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from florch import ModelTrainer, TrainingConfig, TrainingJob


# -- Deterministic data --


def sample_generator(seed=0, input_dim=32, num_classes=10):
    """Yield (x, y) tuples deterministically forever."""
    rng = np.random.default_rng(seed)
    while True:
        x = rng.standard_normal(input_dim).astype(np.float32)
        y = int(rng.integers(0, num_classes))
        yield x, y


def batched_loader(batch_size, seed=0, skip_batches=0, input_dim=32, num_classes=10):
    """Yield batched dicts of tensors from the deterministic sample generator."""
    gen = sample_generator(seed=seed, input_dim=input_dim, num_classes=num_classes)
    # Skip past already-consumed batches for resume
    for _ in range(skip_batches * batch_size):
        next(gen)
    while True:
        xs, ys = [], []
        for _ in range(batch_size):
            x, y = next(gen)
            xs.append(x)
            ys.append(y)
        yield {
            'image': torch.from_numpy(np.stack(xs)),
            'label': torch.tensor(ys, dtype=torch.long),
        }


# -- Simple model (no batch-dependent ops) --


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class SimpleTrainer(ModelTrainer):
    def forward_train(self, inps):
        logits = self.model(inps['image'])
        return {'logits': logits}

    def compute_losses(self, inps, preds):
        loss = F.cross_entropy(preds['logits'], inps['label'])
        return {'loss': loss}


# -- Training jobs --


class VerifyJob(TrainingJob):
    """Uses manual batched_loader with skip for resume."""

    def __init__(self, config, batch_size=32, input_dim=32, num_classes=10):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.num_classes = num_classes
        super().__init__(config)

    def build_data(self):
        skip = self._n_completed_steps_at_start * self.config.grad_accum_steps
        loader = batched_loader(
            batch_size=self.batch_size,
            seed=self.config.seed,
            skip_batches=skip,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )
        return loader, None, 0

    def build_model(self):
        return SimpleMLP(self.input_dim, num_classes=self.num_classes)

    def build_trainer(self, model):
        return SimpleTrainer(model)

    def build_optimizer(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        return optimizer, [lambda step: 1.0]


def load_sample(index, *, rng, group_rng):
    """Stream-compatible load function for synthetic data."""
    x = rng.standard_normal(32).astype(np.float32)
    label = int(rng.integers(0, 10))
    return dict(image=x, label=label)


class StreamVerifyJob(TrainingJob):
    """Uses stream-based data loading with build_stream/stream_to_torch_loader_train."""

    def __init__(self, config, batch_size=32):
        self.batch_size = batch_size
        super().__init__(config)

    def build_data(self):
        indices = list(range(500))
        stream = self.build_stream(examples=indices, load_fn=load_sample)
        loader = self.stream_to_torch_loader_train(stream, self.batch_size)
        return loader, None, 0

    def build_model(self):
        return SimpleMLP()

    def build_trainer(self, model):
        return SimpleTrainer(model)

    def build_optimizer(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        return optimizer, [lambda step: 1.0]


_compile_default = False


def make_config(steps, checkpoint_dir, checkpoint_period=25, grad_accum_steps=1, **kwargs):
    kwargs.setdefault('compile', _compile_default)
    kwargs.setdefault('parallel_build_data', False)
    return TrainingConfig(
        training_steps=steps,
        grad_accum_steps=grad_accum_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_period=checkpoint_period,
        **kwargs,
    )


def get_state(job):
    """Extract model + optimizer state dicts (on CPU) from a completed job."""
    model_state = {k: v.cpu().clone() for k, v in job.model.state_dict().items()}
    opt_state = job.optimizer.state_dict()
    return model_state, opt_state


def compare_states(state_a, state_b, label='', atol=0.0):
    """Compare two model state dicts for equality (exact or within tolerance)."""
    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    assert keys_a == keys_b, f'{label}: key mismatch: {keys_a ^ keys_b}'
    for key in state_a:
        a, b = state_a[key], state_b[key]
        if atol > 0:
            if not torch.allclose(a, b, atol=atol, rtol=0):
                max_diff = (a - b).abs().max().item()
                raise AssertionError(f'{label}: mismatch at {key}, max diff = {max_diff}')
        elif not torch.equal(a, b):
            max_diff = (a - b).abs().max().item()
            raise AssertionError(f'{label}: mismatch at {key}, max diff = {max_diff}')


# -- Tests --


def test_checkpoint_roundtrip():
    """Train, save checkpoint, load it back, verify model state matches."""
    tmpdir = tempfile.mkdtemp(prefix='florch_test_ckpt_')
    try:
        config = make_config(steps=50, checkpoint_dir=tmpdir, checkpoint_period=25)
        job = VerifyJob(config)
        job.train()

        # Model state after training
        trained_state = {k: v.cpu().clone() for k, v in job.model.state_dict().items()}

        # Load the final checkpoint (saved by train()'s finally block)
        ckpt = torch.load(f'{tmpdir}/ckpt-0000050.pth', weights_only=False, map_location='cpu')
        compare_states(trained_state, ckpt['model_state_dict'], 'checkpoint roundtrip')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_resume_equality():
    """Continuous 100 steps must equal 50 steps + resume + 50 more steps."""
    tmpdir_a = tempfile.mkdtemp(prefix='florch_test_resume_a_')
    tmpdir_b = tempfile.mkdtemp(prefix='florch_test_resume_b_')
    try:
        # Run A: 100 steps continuously
        config_a = make_config(steps=100, checkpoint_dir=tmpdir_a)
        job_a = VerifyJob(config_a)
        job_a.train()
        state_a, _ = get_state(job_a)

        # Run B part 1: 50 steps
        config_b1 = make_config(steps=50, checkpoint_dir=tmpdir_b, checkpoint_period=50)
        job_b1 = VerifyJob(config_b1)
        job_b1.train()

        # Run B part 2: resume from checkpoint, train to step 100
        config_b2 = make_config(steps=100, checkpoint_dir=tmpdir_b, checkpoint_period=50)
        job_b2 = VerifyJob(config_b2)
        job_b2.train()
        state_b, _ = get_state(job_b2)

        compare_states(state_a, state_b, 'resume equality')
    finally:
        shutil.rmtree(tmpdir_a, ignore_errors=True)
        shutil.rmtree(tmpdir_b, ignore_errors=True)


def test_grad_accum_equivalence():
    """batch=32,accum=1 must equal batch=8,accum=4."""
    tmpdir_a = tempfile.mkdtemp(prefix='florch_test_accum_a_')
    tmpdir_b = tempfile.mkdtemp(prefix='florch_test_accum_b_')
    try:
        # Run A: large batch, no accumulation
        config_a = make_config(steps=50, checkpoint_dir=tmpdir_a, grad_accum_steps=1)
        job_a = VerifyJob(config_a, batch_size=32)
        job_a.train()
        state_a, _ = get_state(job_a)

        # Run B: small batch, 4x accumulation (same effective batch)
        config_b = make_config(steps=50, checkpoint_dir=tmpdir_b, grad_accum_steps=4)
        job_b = VerifyJob(config_b, batch_size=8)
        job_b.train()
        state_b, _ = get_state(job_b)

        # Allow tiny float32 tolerance: accumulating gradients across 4 backward
        # passes vs 1 introduces rounding differences at ~1e-8 level.
        compare_states(state_a, state_b, 'grad accum equivalence', atol=1e-6)
    finally:
        shutil.rmtree(tmpdir_a, ignore_errors=True)
        shutil.rmtree(tmpdir_b, ignore_errors=True)


def test_stream_resume_equality():
    """Stream-based resume: continuous 100 steps must equal 50 + resume + 50."""
    tmpdir_a = tempfile.mkdtemp(prefix='florch_test_stream_a_')
    tmpdir_b = tempfile.mkdtemp(prefix='florch_test_stream_b_')
    try:
        # Run A: 100 steps continuously
        config_a = make_config(steps=100, checkpoint_dir=tmpdir_a)
        job_a = StreamVerifyJob(config_a)
        job_a.train()
        state_a, _ = get_state(job_a)

        # Run B part 1: 50 steps
        config_b1 = make_config(steps=50, checkpoint_dir=tmpdir_b, checkpoint_period=50)
        job_b1 = StreamVerifyJob(config_b1)
        job_b1.train()

        # Run B part 2: resume from checkpoint, train to step 100
        config_b2 = make_config(steps=100, checkpoint_dir=tmpdir_b, checkpoint_period=50)
        job_b2 = StreamVerifyJob(config_b2)
        job_b2.train()
        state_b, _ = get_state(job_b2)

        compare_states(state_a, state_b, 'stream resume equality')
    finally:
        shutil.rmtree(tmpdir_a, ignore_errors=True)
        shutil.rmtree(tmpdir_b, ignore_errors=True)


def test_ddp_smoke():
    """Launch training with torchrun to verify DDP works."""
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f'  Skipping DDP test (need 2+ GPUs, have {gpu_count})')
        return

    # Launch a short DDP training run
    tmpdir = tempfile.mkdtemp(prefix='florch_test_ddp_')
    try:
        script = f"""\
import sys; sys.path.insert(0, '.')
import tempfile
from examples.verify_features import VerifyJob, make_config
config = make_config(steps=20, checkpoint_dir='{tmpdir}', multi_gpu=True)
job = VerifyJob(config)
job.train()
print('DDP OK')
"""
        result = subprocess.run(
            ['torchrun', '--nproc_per_node=2', '-c', script],
            capture_output=True, text=True, timeout=120,
            cwd='/home/sarandi/rwth-home2/pose/pycharm/florch',
        )
        if result.returncode != 0:
            raise AssertionError(f'DDP failed:\nstdout: {result.stdout}\nstderr: {result.stderr}')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# -- Main --


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddp', action='store_true', help='Also run DDP smoke test')
    parser.add_argument('--compile', action='store_true', help='Run tests with torch.compile')
    args = parser.parse_args()

    if args.compile:
        # Override make_config default
        global _compile_default
        _compile_default = True
        print('Running with compile=True')

    tests = [
        ('checkpoint_roundtrip', test_checkpoint_roundtrip),
        ('resume_equality', test_resume_equality),
        ('stream_resume_equality', test_stream_resume_equality),
        ('grad_accum_equivalence', test_grad_accum_equivalence),
    ]
    if args.ddp:
        tests.append(('ddp_smoke', test_ddp_smoke))

    all_passed = True
    for name, test_fn in tests:
        try:
            test_fn()
            print(f'  PASS: {name}')
        except Exception as e:
            print(f'  FAIL: {name}: {e}')
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
