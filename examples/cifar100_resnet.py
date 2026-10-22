"""CIFAR-100 training with ResNet18 and florch.

Usage:
    # Create the barecat archives first
    python examples/create_cifar100_barecat.py

    # Train with wandb logging
    python examples/cifar100_resnet.py --wandb-project florch-cifar100

    # Train without wandb
    python examples/cifar100_resnet.py --no-wandb

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=2 examples/cifar100_resnet.py --multi-gpu

    # FSDP (single or multi-GPU)
    python examples/cifar100_resnet.py --fsdp
    torchrun --nproc_per_node=2 examples/cifar100_resnet.py --fsdp --multi-gpu
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from barecat import Barecat, DecodedView

from florch import BarecatReader, ModelTrainer, TrainingConfig, TrainingJob


# -- Model --


def resnet18_cifar(num_classes=100):
    """ResNet18 adapted for 32x32 CIFAR images."""
    model = torchvision.models.resnet18(num_classes=num_classes)
    # Replace 7x7 stride-2 conv with 3x3 stride-1 for small images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool (destroys too much spatial info at 32x32)
    model.maxpool = nn.Identity()
    return model


# -- Trainer --


class CIFAR100Trainer(ModelTrainer):
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

DATA_DIR = '/var/tmp/cifar100_data'
ANNO_PATH = f'{DATA_DIR}/cifar100.barecat'
IMG_PATH = f'{DATA_DIR}/cifar100_images.barecat'

MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)


def load_cifar100_item(key, train, anno_reader, img_reader, *, rng, group_rng):
    """Load a single CIFAR-100 item from barecat with augmentation."""
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
    if not os.path.exists(ANNO_PATH) or os.path.exists(IMG_PATH):
        raise FileNotFoundError(
            f'CIFAR-100 barecat not found at {ANNO_PATH}. '
            f'Create it first: python examples/create_cifar100_barecat.py'
        )


# -- Job --


class CIFAR100Job(TrainingJob):
    def __init__(self, config, batch_size=128, dtype=None, warmup_steps=500, lr=1e-3):
        self.batch_size = batch_size
        self.dtype = dtype
        self.warmup_steps = warmup_steps
        self.lr = lr
        super().__init__(config)

    def get_fsdp_wrap_policy(self):
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torchvision.models.resnet import BasicBlock
        return ModuleWrapPolicy({BasicBlock})

    def build_data(self):
        _check_barecat_exists()
        anno_reader = BarecatReader(ANNO_PATH)
        img_reader = BarecatReader(IMG_PATH)
        with Barecat(ANNO_PATH, readonly=True) as bc:
            keys = sorted(bc.globfiles('train/*.msgpack'))
        stream = self.build_stream(
            examples=keys, load_fn=load_cifar100_item,
            extra_args=(True, anno_reader, img_reader),
        )
        loader = self.stream_to_torch_loader_train(stream, self.batch_size)
        return loader, None, 0

    def build_model(self):
        return resnet18_cifar(num_classes=100)

    def build_trainer(self, model):
        return CIFAR100Trainer(model, dtype=self.dtype)

    def build_optimizer(self):
        decay_params = []
        no_decay_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith('.bias') or 'bn' in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': 1e-2},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ],
            lr=self.lr,
        )

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
    parser = argparse.ArgumentParser(description='CIFAR-100 ResNet18 training with florch')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--logdir', type=str, default='/var/tmp/cifar100_florch')
    parser.add_argument('--wandb-project', type=str, default='florch-cifar100')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--fsdp', action='store_true')
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()

    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(args.dtype)
    wandb_project = None if args.no_wandb else args.wandb_project

    config = TrainingConfig(
        training_steps=args.steps,
        grad_accum_steps=args.grad_accum,
        logdir=args.logdir,
        multi_gpu=args.multi_gpu,
        use_fsdp=args.fsdp,
        compile=not args.no_compile,
        clip_grad_norm_factor=3.0,
        loss_scale=128 if dtype == torch.float16 else 1,
        wandb_project=wandb_project,
    )

    job = CIFAR100Job(
        config, batch_size=args.batch_size, dtype=dtype,
        warmup_steps=args.warmup_steps, lr=args.lr,
    )
    job.train()


if __name__ == '__main__':
    main()
