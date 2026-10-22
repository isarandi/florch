"""Core training loop implementation."""

import collections
import itertools
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simplepyutils import logger

from .distributed import sync_metric
from ..exceptions import UnexpectedValueDuringTrainingError
from ..util.misc import smootherstep, to_device_nested


class Timer:
    """Timer for tracking training speed."""

    def __init__(self, maxlen: int = 20):
        self.timestamps = collections.deque([time.perf_counter()], maxlen=maxlen + 1)

    def update_and_get_speed(self) -> float:
        """Record timestamp and return steps per second."""
        self.timestamps.append(time.perf_counter())
        timespan = self.timestamps[-1] - self.timestamps[0]
        done_items = len(self.timestamps) - 1
        return np.float32(done_items / timespan)


def make_norm_hook(norm_losses: List[torch.Tensor]):
    """Create forward hook for norm layer regularization."""
    def hook(module, input, output):
        inp = input[0]
        out = output
        if hasattr(module, 'affine') and module.affine:
            out = (out - module.bias.view(1, -1, 1, 1)) / module.weight.view(1, -1, 1, 1)
        elif hasattr(module, 'elementwise_affine') and module.elementwise_affine:
            out = (out - module.bias) / module.weight
        loss = F.mse_loss(inp, out.detach(), reduction='mean')
        norm_losses.append(loss)

    return hook


def fit_epochless(job):
    """Run the training loop without epoch boundaries.

    Args:
        job: TrainingJob instance with all components built.
    """
    config = job.config
    trainer = job.trainer
    optimizer = job.optimizer
    lr_scheduler = job.lr_scheduler
    grad_clipper = job.grad_clipper
    data_train = job.data_train
    callbacks = job.callbacks
    device = job.device
    rank = job.rank
    world_size = job.world_size
    ddp_trainer = job.ddp_trainer

    initial_substep = int(trainer._train_counter)
    substeps = config.stop_step * config.grad_accum_steps

    scaler = torch.amp.GradScaler(
        init_scale=config.loss_scale * config.grad_accum_steps,
        growth_factor=1 + 1e-10,
        backoff_factor=1 - 1e-10,
        growth_interval=int(1e8),
        enabled=config.loss_scale != 1,
    )

    accum_metrics = {}
    timer = Timer()

    # Per-substep torch seeds are derived STATELESSLY from (seed, substep) so that a
    # resumed run reproduces the exact seed of every substep. The previous scheme
    # (PCG64 chain + advance(initial_substep)) never aligned on resume: numpy's
    # bounded-integer sampling consumes a variable number of generator state steps
    # (32-bit buffering + rejection), so advance(k) does not correspond to k draws.
    def substep_seed(i_substep):
        return int(np.random.SeedSequence((config.seed, i_substep)).generate_state(1)[0])

    if ddp_trainer is None:
        ddp_trainer = trainer

    has_no_sync = hasattr(ddp_trainer, 'no_sync')

    # Norm loss hooks
    norm_losses = []
    handles = []
    if config.norm_loss_factor:
        for m in job.model.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
                handles.append(m.register_forward_hook(make_norm_hook(norm_losses)))

    prepare_inputs = trainer._prepare_inputs
    if config.compile:
        compute_metrics = torch.compile(trainer._compute_metrics)
    else:
        compute_metrics = trainer._compute_metrics

    # Notify callbacks
    for cb in callbacks:
        cb.job = job
        cb.device = device
        if cb.needs_to_run_on_all_devices() or rank == 0:
            cb.on_train_begin(initial_substep // config.grad_accum_steps)

    torch.cuda.empty_cache()

    data_iter = iter(itertools.islice(data_train, substeps - initial_substep))
    for i_substep in itertools.count(initial_substep):
        # Seed BEFORE drawing data: the batch of substep k must be generated under
        # seed k, otherwise the first batch after a resume is drawn under whatever
        # RNG state the fresh process happens to have.
        iter_seed = substep_seed(i_substep)
        torch.random.manual_seed(iter_seed)
        torch.cuda.manual_seed_all(iter_seed)

        try:
            inps = next(data_iter)
        except StopIteration:
            break

        step = i_substep // config.grad_accum_steps

        if i_substep % config.grad_accum_steps == 0:
            for cb in callbacks:
                if cb.needs_to_run_on_all_devices() or rank == 0:
                    cb.on_train_batch_begin(step)

        inps = to_device_nested(inps, device)

        trainer.train(not job.train_in_inference_mode)
        norm_losses.clear()
        inps = prepare_inputs(inps)
        preds, losses = ddp_trainer(inps)

        # Norm loss regularization
        if config.norm_loss_factor and norm_losses:
            losses['norm_loss'] = sum(norm_losses) / len(norm_losses)
            norm_loss_weight = (
                smootherstep(step, config.norm_loss_start_step, config.norm_loss_ramp_steps)
                * config.norm_loss_factor
            )
            lossval = (
                losses["loss"] + norm_loss_weight * torch.nan_to_num(losses['norm_loss'])
            ) / config.grad_accum_steps
        else:
            lossval = losses["loss"] / config.grad_accum_steps

        if not torch.isfinite(lossval) or lossval == 0:
            raise UnexpectedValueDuringTrainingError('Loss', lossval)

        # Backward with optional no_sync for grad accumulation
        is_accumulation_step = (
            i_substep % config.grad_accum_steps != config.grad_accum_steps - 1
        )
        if has_no_sync and is_accumulation_step:
            with ddp_trainer.no_sync():
                scaler.scale(lossval).backward()
        else:
            scaler.scale(lossval).backward()

        # Optimizer step
        grad_norms = []
        if i_substep % config.grad_accum_steps == config.grad_accum_steps - 1:
            if grad_clipper.enabled:
                scaler.unscale_(optimizer)
                param_groups = job.get_grad_clip_param_groups()
                grad_norms, grad_norm_factors = grad_clipper.clip(*param_groups)
                # NaN factors (non-finite gradients) must skip, not step.
                do_step = all(f < grad_clipper.skip_factor for f in grad_norm_factors)
            else:
                do_step = True

            if do_step:
                scaler.step(optimizer)
            elif rank == 0:
                logger.warning(
                    f"Skipping step {step} due to high gradients."
                )

            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        trainer._train_counter += 1

        # Compute and log metrics
        with torch.inference_mode():
            metrics = compute_metrics(inps, preds)
            metrics.update(losses)

            for key, value in metrics.items():
                if key not in accum_metrics:
                    accum_metrics[key] = 0.0
                accum_metrics[key] += value / config.grad_accum_steps

            if i_substep % config.grad_accum_steps == config.grad_accum_steps - 1:
                synced_metrics = {
                    key: sync_metric(value, world_size)
                    for key, value in accum_metrics.items()
                }

                if len(optimizer.param_groups) == 1:
                    synced_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
                else:
                    for pg_idx, pg in enumerate(optimizer.param_groups):
                        synced_metrics[f"learning_rate_{pg_idx}"] = pg["lr"]

                synced_metrics["step_per_sec"] = timer.update_and_get_speed()
                synced_metrics["loss_scale"] = scaler.get_scale()

                for gi, gn in enumerate(grad_norms):
                    synced_metrics[f"gradnorm_{gi}"] = gn

                for cb in callbacks:
                    if cb.needs_to_run_on_all_devices() or rank == 0:
                        cb.on_train_batch_end(step, synced_metrics)

                accum_metrics.clear()

    # Train end
    logger.info('Loop: finished.')
    # substeps is the absolute endpoint (stop_step * grad_accum_steps), not a count
    # of remaining substeps - adding initial_substep would double-count on resume.
    final_step = (substeps - 1) // config.grad_accum_steps
    for cb in callbacks:
        if cb.needs_to_run_on_all_devices() or rank == 0:
            cb.on_train_end(final_step)

    for handle in handles:
        handle.remove()
