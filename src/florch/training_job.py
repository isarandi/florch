import abc
import collections
import functools
import itertools
import os.path as osp
import re
import threading
import time
from typing import Optional, TYPE_CHECKING, Union

import florch.callbacks
import florch.checkpointing
import florch.layers
import florch.parallel_map2
import more_itertools
import numpy as np
import torch
import torch.amp
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from florch.util.distribute_batch import distribute_batch
from florch.util.misc import to_device_nested
from simplepyutils import logger
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

if TYPE_CHECKING:
    from florch import ModelTrainer

import florch.exceptions

class TrainingJob:
    """Encapsulates the input pipeline, model construction, checkpointing and training."""

    def __init__(
        self,
        wandb_project,
        wandb_config,
        logdir,
        init_path,
        load_path,
        training_steps,
        stop_step,
        grad_accum_steps,
        loss_scale,
        dynamic_loss_scale,
        ema_momentum,
        finetune_in_inference_mode,
        validate_period,
        checkpoint_dir,
        checkpoint_period,
        multi_gpu,
        seed,
        n_completed_steps=None,
        workers=None,
        parallel_build_data=True,
        use_fsdp=False,
        clip_grad_norm_median_factor=None,
        clip_grad_norm_quantile=0.5,
        clip_grad_norm_histsize=100,
        norm_loss_factor=0.0,
        norm_loss_start_step=1000,
        norm_loss_ramp_steps=4000,
    ):
        # SETTINGS
        self.use_fsdp = use_fsdp
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config

        self.logdir = logdir

        self.init_path = init_path
        self.load_path = load_path

        self.training_steps = training_steps
        self.stop_step = stop_step
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm_median_factor = clip_grad_norm_median_factor
        self.clip_grad_norm_quantile = clip_grad_norm_quantile
        self.clip_grad_norm_histsize = clip_grad_norm_histsize
        self.loss_scale = loss_scale
        self.dynamic_loss_scale = dynamic_loss_scale
        self.ema_momentum = ema_momentum
        self.finetune_in_inference_mode = finetune_in_inference_mode
        self.norm_loss_factor = norm_loss_factor
        self.norm_loss_start_step = norm_loss_start_step
        self.norm_loss_ramp_steps = norm_loss_ramp_steps

        self.validate_period = validate_period

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_period = checkpoint_period

        self.multi_gpu = multi_gpu
        self.seed = seed
        self.parallel_build_data = parallel_build_data

        # INITIALIZATION

        if n_completed_steps is not None:
            self._n_completed_steps_at_start = n_completed_steps
        else:
            self._n_completed_steps_at_start = self._get_n_completed_steps()

        self.n_workers = workers
        # These will be filled in while building in self._build()
        self.data_train = None
        self.data_val = None
        self.validation_steps = None
        self.model = None
        self.trainer: Optional['ModelTrainer'] = None
        self.ckpt_manager = None
        self.callbacks = None
        self.rank = None
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_functions = None
        self.device = None
        self.grad_clipper = None

    def train(self):
        self._build()

        logger.info("Starting fitting...")
        save_on_exit = True
        try:
            self.fit_epochless()
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
        except florch.exceptions.NonfiniteDuringTrainingError:
            logger.error("Training failed due to non-finite loss.")
            save_on_exit = False
        finally:
            try:
                if save_on_exit and self.rank == 0:
                    self.ckpt_manager.save(
                        self.trainer.adjusted_train_counter, check_interval=False
                    )
                    logger.info("Saved final checkpoint.")
            finally:
                if self.multi_gpu:
                    torch.distributed.destroy_process_group()

    def fit_epochless(self):
        substeps = self.stop_step * self.grad_accum_steps
        initial_substep = self._n_completed_steps_at_start * self.grad_accum_steps

        if self.data_val is not None:
            raise NotImplementedError("Validation data is not supported yet")

        # Force the scale to be constant by setting growth and backoff factors close to 1.
        scaler = torch.amp.GradScaler(
            init_scale=self.loss_scale * self.grad_accum_steps,
            growth_factor=1 + 1e-10,
            backoff_factor=1 - 1e-10,
            growth_interval=int(1e8),
            enabled=self.loss_scale != 1,
        )

        accum_metrics = {}

        timer = Timer()  # Timer for logging step_per_sec to wandb

        self.trainer._train_counter = initial_substep
        self.trainer.grad_accum_steps = self.grad_accum_steps

        for cb in self.callbacks:
            cb.trainer = self.trainer
            cb.device = self.device
            if cb.needs_to_run_on_all_devices() or self.rank == 0:
                cb.on_train_begin(initial_substep // self.grad_accum_steps)

        compute_metrics = torch.compile(self.trainer._compute_metrics)

        # Generate reproducible random seeds for each substep so that dropout and other
        # random operations are consistent across substeps.
        torch_seed_gen = florch.parallel_map2.new_rng(
            self.rng_common, advance_delta=initial_substep
        )

        if self.multi_gpu and not self.use_fsdp:
            ddp_trainer = DDP(self.trainer, find_unused_parameters=True)
        else:
            ddp_trainer = self.trainer

        norm_losses = []
        handles = []

        if self.norm_loss_factor:
            for m in self.model.modules():
                if isinstance(
                    m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d, florch.layers.BatchRenorm2d)
                ):
                    handles.append(m.register_forward_hook(make_norm_hook(norm_losses)))

        torch.cuda.empty_cache()
        for i_substep, inps in enumerate(
            itertools.islice(self.data_train, substeps - initial_substep), start=initial_substep
        ):
            iter_seed = torch_seed_gen.integers(0, int(1e9))
            torch.random.manual_seed(iter_seed)
            torch.cuda.manual_seed_all(iter_seed)

            if i_substep % self.grad_accum_steps == 0:
                for cb in self.callbacks:
                    if cb.needs_to_run_on_all_devices() or self.rank == 0:
                        cb.on_train_batch_begin(i_substep // self.grad_accum_steps)

            inps = to_device_nested(inps, self.device)

            self.trainer.train(not self.trainer.train_in_inference_mode)

            norm_losses.clear()
            inps = self.trainer._prepare_inputs(inps)
            preds, losses = ddp_trainer(inps)

            if self.norm_loss_factor:
                losses['norm_loss'] = (sum(norm_losses) / len(norm_losses)) if norm_losses else 0.0
                norm_loss_factor = (
                    smootherstep(
                        i_substep // self.grad_accum_steps,
                        self.norm_loss_start_step,
                        length=self.norm_loss_ramp_steps,
                    )
                    * self.norm_loss_factor
                )
                lossval = (
                    losses["loss"] + norm_loss_factor * torch.nan_to_num(losses['norm_loss'])
                ) / self.grad_accum_steps
            else:
                lossval = losses["loss"] / self.grad_accum_steps

            # if the lossval is nonfinite or zero, then we raise exception
            if not torch.isfinite(lossval) or lossval == 0:
                raise florch.exceptions.NonfiniteDuringTrainingError('Loss', lossval)

            scaler.scale(lossval).backward()

            grad_norms = [0.0, 0.0]
            if i_substep % self.grad_accum_steps == self.grad_accum_steps - 1:
                if self.clip_grad_norm_median_factor is not None:
                    scaler.unscale_(self.optimizer)
                    grad_norms, grad_norm_factors = self.grad_clipper.clip(
                        self.model.backbone.parameters(), self.model.heatmap_head.parameters()
                    )
                    do_step = all(f < 100.0 for f in grad_norm_factors)
                else:
                    do_step = True

                if do_step:
                    scaler.step(self.optimizer)
                    scaler.update()
                elif self.rank == 0:
                    logger.warning(
                        f"Skipping step {i_substep // self.grad_accum_steps} due to high gradients."
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.lr_scheduler.step()

            self.trainer._train_counter += 1

            with torch.inference_mode():
                metrics = compute_metrics(inps, preds)
                metrics.update(losses)

                for key, value in metrics.items():
                    if key not in accum_metrics:
                        accum_metrics[key] = 0.0
                    accum_metrics[key] += value / self.grad_accum_steps

                if i_substep % self.grad_accum_steps == self.grad_accum_steps - 1:
                    synced_metrics = {
                        key: sync_metric(value, self.world_size)
                        for key, value in accum_metrics.items()
                    }

                    if len(self.optimizer.param_groups) == 1:
                        synced_metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                    else:
                        for i, pg in enumerate(self.optimizer.param_groups):
                            synced_metrics[f"learning_rate_{i}"] = pg["lr"]

                    synced_metrics["step_per_sec"] = timer.update_and_get_speed()
                    synced_metrics["loss_scale"] = scaler.get_scale()
                    synced_metrics["gradnorm_backbone"] = grad_norms[0]
                    synced_metrics["gradnorm_head"] = grad_norms[1]

                    for cb in self.callbacks:
                        if cb.needs_to_run_on_all_devices() or self.rank == 0:
                            cb.on_train_batch_end(
                                i_substep // self.grad_accum_steps, synced_metrics
                            )

                    accum_metrics.clear()

        for cb in self.callbacks:
            if cb.needs_to_run_on_all_devices() or self.rank == 0:
                cb.on_train_end(i_substep // self.grad_accum_steps)

        for handle in handles:
            handle.remove()

    def _build(self):
        self._build_init()

        if self.parallel_build_data:
            thread = threading.Thread(target=self._build_data, daemon=True)
            thread.start()
        else:
            self._build_data()

        self._build_trainer()
        self._build_checkpoint_manager()
        self._build_callbacks()
        self._restore_if_ckpt_available()
        self._build_lr_scheduler()

        if self.parallel_build_data:
            thread.join()

    def _build_init(self):
        if self.multi_gpu:
            torch.distributed.init_process_group("nccl")
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            self.device = torch.device(f"cuda:{torch.distributed.get_node_local_rank()}")
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda")

        self.rng = np.random.Generator(np.random.PCG64(abs(hash((self.seed, self.rank)))))
        self.rng_common = np.random.Generator(np.random.PCG64(abs(hash((self.seed, 0)))))

        self.grad_clipper = AdaptiveGradientClipper(
            self.clip_grad_norm_quantile,
            self.clip_grad_norm_histsize,
            self.clip_grad_norm_median_factor,
        )

    def _build_lr_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=self.lr_functions,
            last_epoch=self._n_completed_steps_at_start - 1,
        )

    def _build_data(self):
        logger.info("Building data...")
        self.data_train, self.data_val, self.validation_steps = self.build_data()
        self.data_train = more_itertools.peekable(self.data_train)
        self.data_train.peek()
        logger.info("Data built.")

    def _build_trainer(self):
        logger.info("Building trainer...")
        seed = abs(hash((self.seed,)))
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.model is None:
            model = self.build_model()
        else:
            model = self.model
        self.trainer = torch.compile(self.build_trainer(model).to(self.device))
        self.model = self.trainer.model

        self.optimizer, self.lr_functions = self.build_optimizer()

        if self.multi_gpu and self.use_fsdp:

            vision_model = self.model.backbone[1].model
            NestedTensorBlock = type(vision_model.blocks[0])
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls={NestedTensorBlock}
            )
            ddp_trainer = FSDP(
                self.trainer,
                auto_wrap_policy=wrap_policy,
                use_orig_params=True,
            )
            self.trainer = ddp_trainer
            self.model = self.trainer.model

        logger.info("Trainer built.")

    def _build_model(self):
        logger.info("Building model...")
        model = self.build_model()
        logger.info("Model built.")
        return model

    def _build_callbacks(self):
        cbacks = [
            florch.callbacks.Checkpoint(self.ckpt_manager),
            florch.callbacks.Progbar(self.training_steps),
        ]

        if self.wandb_project:
            cbacks.append(
                florch.callbacks.Wandb(
                    project_name=self.wandb_project,
                    logdir=self.logdir,
                    config_dict=self.wandb_config,
                )
            )

        if self.multi_gpu and florch.callbacks.SyncBNStats.has_any_bn_layers(self.model):
            cbacks.append(florch.callbacks.SyncBNStats())

        if self.finetune_in_inference_mode:
            switch_step = (
                self.training_steps - self.finetune_in_inference_mode
            ) * self.grad_accum_steps
            cbacks.append(
                florch.callbacks.SwitchToInferenceModeCallback(switch_step, self.ckpt_manager)
            )
        self.callbacks = cbacks + list(self.build_callbacks())

    @abc.abstractmethod
    def build_data(self):
        pass

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def build_trainer(self, model):
        pass

    @abc.abstractmethod
    def build_optimizer(self):
        pass

    def build_callbacks(self):
        return []

    def _build_checkpoint_manager(self):
        self.ckpt_manager = florch.checkpointing.CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            grad_clipper=self.grad_clipper,
            directory=self.checkpoint_dir,
            max_to_keep=2,
            checkpoint_interval=self.checkpoint_period,
            fsdp=self.multi_gpu,
        )

    def _get_n_completed_steps(self):
        load_path = self.get_load_path()
        if load_path is not None and load_path != self.init_path:
            return get_step_count_from_checkpoint_path(load_path)
        else:
            return 0

    def get_load_path(self):
        if self.load_path:
            return self.load_path

        latest = florch.checkpointing.latest_checkpoint(self.checkpoint_dir)
        if latest is not None:
            return latest

        return self.init_path

    def _restore_if_ckpt_available(self):
        load_path = self.get_load_path()
        if load_path:
            load_path = ensure_absolute_path(load_path, self.checkpoint_dir)
            logger.info(f"Restoring from {load_path}...")
            checkpoint = torch.load(load_path, weights_only=False, map_location=self.device)

            if self.multi_gpu and self.use_fsdp:
                with FSDP.state_dict_type(
                    self.model,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=StateDictConfig(offload_to_cpu=True),
                ):
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False
                    )
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    checkpoint['model_state_dict'], strict=False
                )

            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            if self.optimizer is not None:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    logger.warning("Failed to load optimizer state dict.")

            if self.grad_clipper is not None:
                if 'grad_clipper_state_dict' in checkpoint:
                    self.grad_clipper.load_state_dict(checkpoint['grad_clipper_state_dict'])
                else:
                    grad_clipper_state_dict = {
                        'histories': [[0.7] * self.clip_grad_norm_histsize] * 2
                    }
                    self.grad_clipper.load_state_dict(grad_clipper_state_dict)

            del checkpoint

    def restore(self):
        if self.model is None:
            self.model = self._build_model()
        self._restore_if_ckpt_available()

    def build_stream(
        self, examples, load_fn, extra_args, shuffle_initially=True, shuffle_before_each_epoch=True
    ):
        return _build_stream(
            examples,
            load_fn,
            extra_args,
            shuffle_initially,
            shuffle_before_each_epoch,
            rng=florch.parallel_map2.new_rng(self.rng),
        )

    def build_roundrobin_stream(
        self,
        example_sections,
        load_fn,
        extra_args,
        batch_size,
        roundrobin_sizes,
        rewrap=True,
        shuffle_before_each_epoch=True,
    ):
        if rewrap:
            roundrobin_sizes = distribute_batch(
                roundrobin_sizes,
                batch_size // self.grad_accum_steps,
                4 * self.grad_accum_steps,
            )
        return _build_roundrobin_stream(
            example_sections,
            load_fn,
            extra_args,
            roundrobin_sizes,
            shuffle_before_each_epoch,
            rng=florch.parallel_map2.new_rng(self.rng),
        )

    def merge_streams(self, streams, batch_sizes):
        for b in batch_sizes:
            assert b % self.grad_accum_steps == 0

        return florch.parallel_map2.roundrobin(
            streams, [b // self.grad_accum_steps for b in batch_sizes]
        )

    def merge_streams_to_torch_loader_train(self, streams, batch_sizes):
        merged_stream = self.merge_streams(streams, batch_sizes)
        return self.stream_to_torch_loader_train(merged_stream, sum(batch_sizes))

    def stream_to_torch_loader_train(self, stream, batch_size):
        return _stream_to_batched_torch_loader(
            stream,
            batch_size=batch_size // self.grad_accum_steps,
            n_completed_batches=self._n_completed_steps_at_start * self.grad_accum_steps,
            n_total_batches=self.training_steps * self.grad_accum_steps,
            rank=self.rank,
            world_size=self.world_size,
        )

    def stream_to_torch_loader_test(self, stream, batch_size):
        return _stream_to_batched_torch_loader(
            stream,
            batch_size=batch_size,
            n_completed_batches=0,
            n_total_batches=None,
            n_workers=self.n_workers,
            rank=self.rank,
            world_size=self.world_size,
        )


def _stream_to_batched_torch_loader(
    stream,
    batch_size,
    n_completed_batches,
    n_total_batches=None,
    n_workers=None,
    rank=0,
    world_size=1,
):
    n_completed_items = n_completed_batches * batch_size
    n_total_items = n_total_batches * batch_size if n_total_batches is not None else None
    sliced_stream = itertools.islice(
        stream, n_completed_items + rank, n_total_items, world_size
    )
    assert batch_size % world_size == 0, "Batch size must be divisible by the number of devices"
    return florch.parallel_map2.function_calls_to_batched_torch_loader(
        sliced_stream,
        batch_size // world_size,
        n_workers=n_workers,
    )


def _build_stream(
    examples, load_fn, extra_args, shuffle_initially, shuffle_before_each_epoch, rng
):
    shuffler_rng = florch.parallel_map2.new_rng(rng)
    loader_rng = florch.parallel_map2.new_rng(rng)
    item_stream = florch.parallel_map2.iterate_repeatedly(
        examples, shuffle_initially, shuffle_before_each_epoch, shuffler_rng
    )
    return _build_fns_args_kwargs_stream(item_stream, load_fn, extra_args, rng=loader_rng)


def _build_roundrobin_stream(
    example_sections, load_fn, extra_args, roundrobin_sizes, shuffle_before_each_epoch, rng
):
    item_streams = [
        florch.parallel_map2.iterate_repeatedly(
            examples, shuffle_before_each_epoch, florch.parallel_map2.new_rng(rng)
        )
        for examples in example_sections
    ]

    fns_args_kwargs_streams = [
        _build_fns_args_kwargs_stream(
            item_stream, load_fn, extra_args, rng=florch.parallel_map2.new_rng(rng)
        )
        for item_stream in item_streams
    ]

    return florch.parallel_map2.roundrobin(fns_args_kwargs_streams, roundrobin_sizes)


def _build_fns_args_kwargs_stream(items, load_fn, extra_args, rng):
    for item in items:
        yield load_fn, (item, *extra_args), dict(rng=florch.parallel_map2.new_rng(rng))


def get_step_count_from_checkpoint_path(checkpoint_path):
    return int(re.search(r"ckpt-(?P<num>\d+)\.pth", checkpoint_path)["num"])


def ensure_absolute_path(path, root):
    if not root:
        return path

    if osp.isabs(path):
        return path
    else:
        return osp.join(root, path)


class Timer:
    def __init__(self, maxlen=20):
        self.timestamps = collections.deque([time.perf_counter()], maxlen=maxlen + 1)

    def update_and_get_speed(self):
        self.timestamps.append(time.perf_counter())
        timespan = self.timestamps[-1] - self.timestamps[0]
        done_items = len(self.timestamps) - 1
        return np.float32(done_items / timespan)


def sync_metric(value: Union[float, torch.Tensor], world_size: int) -> float:
    """Synchronize metric across processes in DDP."""

    if torch.is_tensor(value):
        tensor = value
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            return tensor.item() / world_size
        else:
            return tensor.item()
    else:
        return float(value)


def get_world_size():
    """Get the number of processes in DDP."""
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def make_norm_hook(norm_losses):
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


class AdaptiveGradientClipper:
    def __init__(self, quantile=0.5, size=100, factor=1.0):
        self.factor = factor
        self.size = size
        self.quantile = quantile
        self.histories = None

    def clip(self, *param_groups):
        if self.histories is None:
            self.histories = [collections.deque() for _ in range(len(param_groups))]

        if len(param_groups) != len(self.histories):
            raise ValueError(
                f"Expected {len(self.histories)} parameter groups, but got {len(param_groups)}"
            )

        norms = []
        norm_factors = []
        for hist, param in zip(self.histories, param_groups):
            clip_norm = np.quantile(hist, self.quantile) * self.factor if len(hist) > 0 else np.inf
            norm = torch.nn.utils.clip_grad_norm_(param, max_norm=clip_norm).item()
            hist.append(norm)
            while len(hist) > self.size:
                hist.popleft()

            norms.append(norm)
            norm_factors.append(norm / clip_norm)

        return norms, norm_factors

    def state_dict(self):
        if self.histories is not None:
            return {'histories': [list(hist) for hist in self.histories]}
        else:
            return {'histories': None}

    def load_state_dict(self, state_dict):
        if state_dict['histories'] is None:
            self.histories = None
        else:
            self.histories = [collections.deque(hist) for hist in state_dict['histories']]


def smootherstep(x, x0=0, x1=None, length=None):
    if length is None:
        length = x1 - x0
    y = np.clip((x - x0) / length, 0, 1)
    return y * y * y * (y * (y * 6.0 - 15.0) + 10.0)
