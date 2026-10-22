"""CIFAR-10 training with florch, loading from barecat.

Usage:
    # Create the barecat archives first
    python examples/create_cifar10_barecat.py

    # Single GPU
    python examples/cifar10.py --steps 5000 --logdir /var/tmp/cifar10

    # Mixed precision
    python examples/cifar10.py --steps 5000 --logdir /var/tmp/cifar10 --dtype bfloat16

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 examples/cifar10.py --steps 5000 --logdir /var/tmp/cifar10 --multi-gpu
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from barecat import Barecat, DecodedView

from florch import BarecatReader, ModelTrainer, TrainingConfig, TrainingJob


# -- Model --


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -- Trainer --


class CIFAR10Trainer(ModelTrainer):
    def __init__(self, model, dtype=None):
        super().__init__(model)
        self.dtype = dtype

    def forward_train(self, inps):
        with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.dtype is not None):
            logits = self.model(inps['image'])
        return {'logits': logits}

    def compute_losses(self, inps, preds):
        loss = F.cross_entropy(preds['logits'].float(), inps['label'])
        return {'loss': loss}

    def compute_metrics(self, inps, preds):
        correct = (preds['logits'].argmax(1) == inps['label']).float().mean()
        return {'accuracy': correct}


# -- Data loading --

DATA_DIR = '/var/tmp/cifar10_data'
ANNO_PATH = f'{DATA_DIR}/cifar10.barecat'
IMG_PATH = f'{DATA_DIR}/cifar10_images.barecat'

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def load_cifar10_item(key, train, anno_reader, img_reader, *, rng, group_rng):
    """Load a single CIFAR-10 item from barecat with augmentation."""
    d = DecodedView(anno_reader)[key]
    image = DecodedView(img_reader)[d['impath']]
    image = image.astype(np.float32) / 255.0
    label = d['label']

    if train:
        # Random crop with padding=4
        padded = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='reflect')
        i = rng.integers(0, 9)
        j = rng.integers(0, 9)
        image = padded[i:i + 32, j:j + 32]
        # Random horizontal flip
        if rng.random() > 0.5:
            image = image[:, ::-1]

    image = (image - MEAN) / STD
    image = image.transpose(2, 0, 1)  # CHW
    return dict(image=image, label=label)


def _check_barecat_exists():
    if os.path.exists(ANNO_PATH) and os.path.exists(IMG_PATH):
        return
    raise FileNotFoundError(
        f'CIFAR-10 barecat not found at {ANNO_PATH}. '
        f'Create it first: python examples/create_cifar10_barecat.py'
    )


# -- Job --


class CIFAR10Job(TrainingJob):
    def __init__(self, config, batch_size=128, dtype=None, warmup_steps=200):
        self.batch_size = batch_size
        self.dtype = dtype
        self.warmup_steps = warmup_steps
        super().__init__(config)

    def build_data(self):
        _check_barecat_exists()
        anno_reader = BarecatReader(ANNO_PATH)
        img_reader = BarecatReader(IMG_PATH)
        with Barecat(ANNO_PATH, readonly=True) as bc:
            keys = sorted(bc.globfiles('train/*.msgpack'))
        stream = self.build_stream(
            examples=keys, load_fn=load_cifar10_item,
            extra_args=(True, anno_reader, img_reader),
        )
        loader = self.stream_to_torch_loader_train(stream, self.batch_size)
        return loader, None, 0

    def build_model(self):
        return SmallCNN()

    def build_trainer(self, model):
        return CIFAR10Trainer(model, dtype=self.dtype)

    def build_optimizer(self):
        # Separate decay / no-decay param groups (bias and norm → no weight decay)
        decay_params = []
        no_decay_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith('.bias') or 'norm' in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': 1e-2},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=1e-3,
        )

        # Warmup + cosine decay schedule
        warmup_steps = self.warmup_steps
        total_steps = self.config.training_steps

        def lr_fn(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return optimizer, [lr_fn, lr_fn]


# -- CLI --


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 training with florch')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--logdir', type=str, default='/var/tmp/cifar10_florch')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'],
                        help='Mixed precision dtype for autocast')
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()

    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(args.dtype)

    config = TrainingConfig(
        training_steps=args.steps,
        grad_accum_steps=args.grad_accum,
        checkpoint_dir=f'{args.logdir}/checkpoints',
        checkpoint_period=args.steps // 5 or 1,
        logdir=args.logdir,
        multi_gpu=args.multi_gpu,
        compile=not args.no_compile,
        clip_grad_norm_factor=3.0,
        loss_scale=128 if dtype == torch.float16 else 1,
    )

    job = CIFAR10Job(config, batch_size=args.batch_size, dtype=dtype,
                     warmup_steps=args.warmup_steps)
    job.train()


if __name__ == '__main__':
    main()
