"""Training job orchestration."""

import abc
import os.path as osp
import re
import threading
from typing import List, NamedTuple, Optional, Tuple

import more_itertools
import numpy as np
import torch

from ..config import TrainingConfig
from ..checkpointing import CheckpointManager, latest_checkpoint
from ..grad_clipping import AdaptiveGradientClipper
from ..model_trainer import ModelTrainer
from ..util.misc import new_rng
from .distributed import init_distributed
from .loop import fit_epochless


# Clipping history value for checkpoints that have no clipper state.
INITIAL_GRAD_NORM = 0.7


class DataLoaders(NamedTuple):
    """Return type for build_data()."""
    train: object
    val: object = None
    val_steps: int = 0


class TrainingJob(abc.ABC):
    """Base class for training jobs.

    Subclasses must implement:
    - build_data(): Returns DataLoaders(train, val, val_steps)
    - build_model(): Returns the model
    - build_trainer(model): Returns ModelTrainer wrapping the model
    - build_optimizer(): Returns (optimizer, lr_functions)

    Optionally override:
    - prepare_inputs(inps): Preprocess batch before forward pass
    - get_grad_clip_param_groups(): Return param groups for gradient clipping
    - build_callbacks(): Returns list of additional callbacks
    - get_fsdp_wrap_policy(): Returns FSDP auto wrap policy

    Example:
        class MyJob(TrainingJob):
            def build_data(self):
                stream = DataStream(paths, load_fn, seed=self.config.seed)
                loader = stream_to_loader(stream, micro_batch_size=64)
                return DataLoaders(train=loader)

            def build_model(self):
                return MyModel()

            def build_trainer(self, model):
                return MyTrainer(model)

            def build_optimizer(self):
                opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
                return opt, [lambda step: 1.0]

        config = TrainingConfig(training_steps=100000)
        job = MyJob(config)
        job.train()
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Will be set during _build()
        self.data_train = None
        self.data_val = None
        self.validation_steps = None
        self.model = None
        self.trainer: Optional[ModelTrainer] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_functions = None
        self.ckpt_manager = None
        self.callbacks = None
        self.grad_clipper = None

        # Distributed state
        self.rank = 0
        self.world_size = 1
        self.device = None
        self.ddp_trainer = None

        # RNGs
        self.rng = None
        self.rng_common = None

        # Training loop state
        self.train_in_inference_mode = False

        # Determine completed steps
        if config.n_completed_steps is not None:
            self._n_completed_steps_at_start = config.n_completed_steps
        else:
            self._n_completed_steps_at_start = self._get_n_completed_steps()

    # === Training loop state ===

    @property
    def training_steps(self):
        return self.config.training_steps

    @property
    def adjusted_train_counter(self) -> int:
        """Training step counter adjusted for gradient accumulation."""
        return int(self.trainer.adjusted_train_counter)

    # === Methods called by the training loop ===

    def get_grad_clip_param_groups(self):
        """Return parameter groups for gradient clipping.

        Override to provide model-specific grouping (e.g., backbone vs head).
        Default: all parameters as one group.

        Returns:
            Either a dict mapping a group name to its parameter iterable, e.g.
            ``{'backbone': ..., 'head': ...}`` (gradient norms are then logged as
            ``gradnorm_backbone`` and ``gradnorm_head``), or a list of parameter
            iterables (logged as ``gradnorm_0``, ``gradnorm_1``, ...). Each group has
            its own clipping history.
        """
        return [self.model.parameters()]

    # === Training ===

    def train(self):
        """Run training."""
        from ..exceptions import UnexpectedValueDuringTrainingError
        from simplepyutils import logger

        self._build()

        self.trainer.set_training_state(
            substep=self._n_completed_steps_at_start * self.config.grad_accum_steps,
            grad_accum_steps=self.config.grad_accum_steps,
        )

        logger.info("Starting training...")
        save_on_exit = True

        try:
            fit_epochless(self)
        except KeyboardInterrupt:
            logger.info("Training interrupted.")
        except UnexpectedValueDuringTrainingError:
            logger.error("Training failed due to unexpected value (non-finite or zero loss).")
            save_on_exit = False
        finally:
            try:
                if save_on_exit and self.rank == 0:
                    self.ckpt_manager.save(
                        self.adjusted_train_counter, check_interval=False
                    )
                    logger.info("Saved final checkpoint.")
            finally:
                if self.config.multi_gpu or getattr(self, '_initialized_single_gpu_fsdp', False):
                    torch.distributed.destroy_process_group()

    def _build(self):
        """Build all components."""
        self._build_init()

        if self.config.parallel_build_data:
            thread = threading.Thread(target=self._build_data, daemon=True)
            thread.start()
        else:
            self._build_data()

        self._build_trainer()
        self._build_checkpoint_manager()
        self._build_callbacks()
        self._restore_if_ckpt_available()
        self._build_lr_scheduler()

        if self.config.parallel_build_data:
            thread.join()

    def _build_init(self):
        """Initialize distributed training and RNGs."""
        if self.config.multi_gpu:
            self.rank, self.world_size, self.device = init_distributed()
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # FSDP requires process group even on single GPU
            if self.config.use_fsdp and not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method="tcp://localhost:0",
                    world_size=1,
                    rank=0,
                )
                self._initialized_single_gpu_fsdp = True

        # rng differs per rank (for rank-local randomness); rng_common is identical on
        # all ranks and seeds the data streams, which every rank must traverse in the
        # same order for the per-rank split to partition each global batch.
        self.rng = np.random.Generator(
            np.random.PCG64(abs(hash((self.config.seed, self.rank))))
        )
        self.rng_common = np.random.Generator(
            np.random.PCG64(abs(hash((self.config.seed, 0))))
        )

        self.grad_clipper = AdaptiveGradientClipper(
            quantile=self.config.clip_grad_norm_quantile,
            size=self.config.clip_grad_norm_histsize,
            factor=self.config.clip_grad_norm_factor,
            skip_factor=self.config.clip_grad_norm_skip_factor,
        )

    def _build_data(self):
        """Build data loaders."""
        from simplepyutils import logger
        logger.info("Building data...")
        result = self.build_data()
        if isinstance(result, DataLoaders):
            self.data_train = result.train
            self.data_val = result.val
            self.validation_steps = result.val_steps
        else:
            self.data_train, self.data_val, self.validation_steps = result
        if self.data_val is not None:
            logger.warning(
                "build_data() returned validation data, but the training loop does not "
                "run validation yet; it is ignored."
            )
        self.data_train = more_itertools.peekable(self.data_train)
        self.data_train.peek()
        logger.info("Data built.")

    def _build_trainer(self):
        """Build model and trainer."""
        from simplepyutils import logger

        logger.info("Building trainer...")

        seed = abs(hash((self.config.seed,)))
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.model is None:
            self.model = self.build_model()

        self.trainer = torch.compile(self.build_trainer(self.model).to(self.device)) if self.config.compile else self.build_trainer(self.model).to(self.device)
        self.model = self.trainer.model

        # FSDP wrapping BEFORE optimizer creation
        if self.config.use_fsdp:
            self._wrap_fsdp()

        if self.config.use_fsdp:
            self.ddp_trainer = self.trainer
        elif self.config.multi_gpu:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.ddp_trainer = DDP(
                self.trainer,
                find_unused_parameters=self.config.find_unused_parameters,
            )
        else:
            self.ddp_trainer = self.trainer

        self.optimizer, self.lr_functions = self.build_optimizer()

        logger.info("Trainer built.")

    def _wrap_fsdp(self):
        """Wrap trainer with FSDP."""
        from simplepyutils import logger
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            BackwardPrefetch,
            CPUOffload,
            MixedPrecision,
        )

        wrap_policy = self.get_fsdp_wrap_policy()
        if wrap_policy is None:
            logger.warning(
                "FSDP enabled but get_fsdp_wrap_policy() returned None. "
                "Override get_fsdp_wrap_policy() to specify which layers to shard."
            )

        sharding_strategy = getattr(ShardingStrategy, self.config.fsdp_sharding_strategy)

        backward_prefetch = None
        if self.config.fsdp_backward_prefetch:
            backward_prefetch = getattr(BackwardPrefetch, self.config.fsdp_backward_prefetch)

        cpu_offload = None
        if self.config.fsdp_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)

        mixed_precision = None
        if self.config.fsdp_mixed_precision:
            if self.config.fsdp_mixed_precision == 'bf16':
                dtype = torch.bfloat16
            elif self.config.fsdp_mixed_precision == 'fp16':
                dtype = torch.float16
            else:
                raise ValueError(
                    f"Unknown fsdp_mixed_precision: {self.config.fsdp_mixed_precision}"
                )
            mixed_precision = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
            )

        self.trainer = FSDP(
            self.trainer,
            auto_wrap_policy=wrap_policy,
            sharding_strategy=sharding_strategy,
            backward_prefetch=backward_prefetch,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
            use_orig_params=self.config.fsdp_use_orig_params,
            device_id=self.device,
        )
        self.model = self.trainer.model
        logger.info(
            f"FSDP wrapped with strategy={self.config.fsdp_sharding_strategy}"
        )

    def get_fsdp_wrap_policy(self):
        """Return FSDP auto wrap policy. Override to specify which layers to shard."""
        return None

    def _build_checkpoint_manager(self):
        """Build checkpoint manager."""
        self.ckpt_manager = CheckpointManager(
            model=self.model,
            optimizer=self.optimizer,
            grad_clipper=self.grad_clipper,
            directory=self.config.checkpoint_dir,
            max_to_keep=self.config.checkpoint_max_to_keep,
            checkpoint_interval=self.config.checkpoint_period,
            fsdp=self.config.use_fsdp,
        )

    def _build_callbacks(self):
        """Build callbacks list."""
        from .. import callbacks as cb_module

        cbacks = [
            cb_module.Checkpoint(self.ckpt_manager),
            cb_module.Progbar(self.config.stop_step),
        ]

        if self.config.wandb_project:
            wandb_config = self.config.wandb_config
            if wandb_config is None:
                import dataclasses
                wandb_config = dataclasses.asdict(self.config)
            cbacks.append(cb_module.Wandb(
                project_name=self.config.wandb_project,
                logdir=self.config.logdir,
                config_dict=wandb_config,
                every_n_steps=30,
            ))

        if self.config.multi_gpu and cb_module.SyncBNStats.has_any_bn_layers(self.model):
            cbacks.append(cb_module.SyncBNStats())

        if self.config.finetune_in_inference_mode:
            switch_step = (
                self.config.training_steps - self.config.finetune_in_inference_mode
            ) * self.config.grad_accum_steps
            cbacks.append(
                cb_module.SwitchToInferenceModeCallback(switch_step, self.ckpt_manager)
            )

        self.callbacks = cbacks + list(self.build_callbacks())

    def _build_lr_scheduler(self):
        """Build LR scheduler."""
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=self.lr_functions,
            last_epoch=self._n_completed_steps_at_start - 1,
        )

    def _get_n_completed_steps(self) -> int:
        """Get number of completed steps from checkpoint."""
        load_path = self.get_load_path()
        if load_path is not None and load_path != self.config.init_path:
            return _get_step_from_checkpoint_path(load_path)
        return 0

    def get_load_path(self) -> Optional[str]:
        """Get path to checkpoint to load."""
        if self.config.load_path:
            return self.config.load_path

        if self.config.checkpoint_dir:
            latest = latest_checkpoint(self.config.checkpoint_dir)
            if latest is not None:
                return latest

        return self.config.init_path

    def _restore_if_ckpt_available(self):
        """Restore from checkpoint if available."""
        from simplepyutils import logger

        load_path = self.get_load_path()
        if not load_path:
            return

        if not osp.isabs(load_path) and self.config.checkpoint_dir:
            load_path = osp.join(self.config.checkpoint_dir, load_path)

        logger.info(f"Restoring from {load_path}...")
        checkpoint = torch.load(load_path, weights_only=False, map_location=self.device)

        state_dict = checkpoint.get('model_state_dict', checkpoint)

        if self.config.use_fsdp:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
            )
            with FSDP.state_dict_type(
                self.trainer,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=False
                ),
            ):
                missing_keys, unexpected_keys = self.trainer.load_state_dict(
                    state_dict, strict=False
                )
        else:
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

        # Load optimizer
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                if self.config.use_fsdp:
                    from torch.distributed.fsdp import (
                        FullyShardedDataParallel as FSDP,
                        FullOptimStateDictConfig,
                        StateDictType,
                    )
                    with FSDP.state_dict_type(
                        self.trainer,
                        state_dict_type=StateDictType.FULL_STATE_DICT,
                        optim_state_dict_config=FullOptimStateDictConfig(
                            offload_to_cpu=True, rank0_only=False
                        ),
                    ):
                        optim_state = FSDP.optim_state_dict_to_load(
                            self.trainer, self.optimizer,
                            checkpoint['optimizer_state_dict']
                        )
                        self.optimizer.load_state_dict(optim_state)
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state dict: {e}")

        # Load gradient clipper. A checkpoint without clipper state (e.g. pretrained
        # weights) starts every group's history at 0.7, so that the first steps after
        # loading are already clipped instead of running unclipped until a history
        # has built up.
        if self.grad_clipper is not None:
            if 'grad_clipper_state_dict' in checkpoint:
                self.grad_clipper.load_state_dict(checkpoint['grad_clipper_state_dict'])
            else:
                n_groups = len(self.get_grad_clip_param_groups())
                self.grad_clipper.load_state_dict({
                    'histories': [
                        [INITIAL_GRAD_NORM] * self.config.clip_grad_norm_histsize
                        for _ in range(n_groups)
                    ]
                })

    def restore(self):
        """Restore model without training (for inference)."""
        self._build_init()
        if self.model is None:
            self.model = self.build_model()
        self._restore_if_ckpt_available()

    # === Abstract methods to implement ===

    @abc.abstractmethod
    def build_data(self):
        """Build data loaders.

        Returns:
            DataLoaders(train, val, val_steps) or a plain tuple.
            val and val_steps can be None/0 if no validation.
        """
        pass

    @abc.abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Build and return the model."""
        pass

    @abc.abstractmethod
    def build_trainer(self, model: torch.nn.Module) -> ModelTrainer:
        """Build and return the trainer wrapping the model."""
        pass

    @abc.abstractmethod
    def build_optimizer(self) -> Tuple[torch.optim.Optimizer, List]:
        """Build optimizer and LR functions.

        Returns:
            Tuple of (optimizer, lr_functions).
            lr_functions is a list of callables, one per param group,
            each taking step and returning LR multiplier.
        """
        pass

    def build_callbacks(self) -> List:
        """Build additional callbacks. Override to add custom callbacks."""
        return []

    # === Stream-based data loading helpers ===

    def build_stream(
        self, examples, load_fn, extra_args=(),
        shuffle_initially=True, shuffle_before_each_epoch=True, group_sample_size=None,
    ):
        """Build a seekable DataStream over examples."""
        from ..streams import DataStream
        seed = int(new_rng(self.rng_common).integers(0, 2**32))
        return DataStream(
            items=examples,
            load_fn=load_fn,
            seed=seed,
            shuffle=shuffle_initially or shuffle_before_each_epoch,
            extra_args=extra_args,
            group_sample_size=group_sample_size,
        )

    def build_roundrobin_stream(
        self, example_sections, load_fn, extra_args=(),
        roundrobin_sizes=None, batch_size=None, group_sample_size=None,
    ):
        """Build a seekable RoundRobinStream over multiple dataset sections.

        Args:
            example_sections: List of dataset sections to round-robin over.
            load_fn: Function to load a single example.
            extra_args: Extra arguments passed to load_fn.
            roundrobin_sizes: Number of examples per section in each round-robin cycle.
            batch_size: This stream's share of the global batch (before splitting for
                gradient accumulation), i.e. the value later passed to merge_streams.
                If given, roundrobin_sizes are relative weights, redistributed so that
                every micro-batch gets nearly the same section mix. If None, the sizes
                are cycled unchanged, which keeps multi-view groups intact.
            group_sample_size: If set, samples are drawn in groups of this size
                (e.g. for multi-view batches).
        """
        from ..streams import RoundRobinStream
        ga = self.config.grad_accum_steps
        seed = int(new_rng(self.rng_common).integers(0, 2**32))
        return RoundRobinStream(
            sections=example_sections,
            sizes=roundrobin_sizes,
            load_fn=load_fn,
            seed=seed,
            extra_args=extra_args,
            micro_batch_size=batch_size // ga if batch_size is not None else None,
            grad_accum_steps=ga,
            group_sample_size=group_sample_size,
        )

    def merge_streams(self, streams, batch_sizes):
        """Merge multiple streams via MultiStream with given batch sizes.

        batch_sizes are the TOTAL batch sizes (before grad accumulation splitting).
        They are divided by grad_accum_steps internally to get per-substep sizes.
        Each batch_size must be divisible by grad_accum_steps.
        """
        from ..streams import MultiStream
        ga = self.config.grad_accum_steps
        for b in batch_sizes:
            assert b % ga == 0, f'batch_size {b} not divisible by grad_accum_steps {ga}'
        seed = int(new_rng(self.rng_common).integers(0, 2**32))
        return MultiStream(streams, [b // ga for b in batch_sizes], seed=seed)

    def stream_to_torch_loader_train(self, stream, batch_size):
        """Convert a stream to a batched torch loader for training.

        batch_size is the GLOBAL batch size per optimizer step: summed over all
        ranks, before splitting for gradient accumulation. It is divided by
        grad_accum_steps and by the number of ranks to get each rank's micro-batch,
        so the same batch_size trains with the same global batch on any GPU count.
        """
        from ..streams import stream_to_loader
        return stream_to_loader(
            stream,
            micro_batch_size=self._per_rank_size(batch_size, self.config.grad_accum_steps),
            num_workers=self.config.workers,
            completed_steps=self._n_completed_steps_at_start,
            total_steps=self.config.training_steps,
            grad_accum_steps=self.config.grad_accum_steps,
            rank=self.rank,
            world_size=self.world_size,
        )

    def stream_to_torch_loader_test(self, stream, batch_size):
        """Convert a stream to a batched torch loader for validation.

        batch_size is global (summed over all ranks), as for training.
        """
        from ..streams import stream_to_loader
        return stream_to_loader(
            stream, micro_batch_size=self._per_rank_size(batch_size, 1),
            num_workers=self.config.workers,
            rank=self.rank, world_size=self.world_size,
        )

    def _per_rank_size(self, batch_size, n_substeps):
        """Split a global batch size into a per-rank, per-substep micro-batch size."""
        divisor = n_substeps * self.world_size
        if batch_size % divisor != 0:
            raise ValueError(
                f'Batch size {batch_size} must be divisible by grad_accum_steps * '
                f'world_size = {n_substeps} * {self.world_size}'
            )
        return batch_size // divisor

    def merge_streams_to_torch_loader_train(self, streams, batch_sizes):
        """Merge streams and convert to a training loader in one call."""
        merged_stream = self.merge_streams(streams, batch_sizes)
        return self.stream_to_torch_loader_train(merged_stream, sum(batch_sizes))


def _get_step_from_checkpoint_path(checkpoint_path):
    """Extract step number from checkpoint filename."""
    match = re.search(r"ckpt-(?P<num>\d+)\.pth", checkpoint_path)
    if match:
        return int(match["num"])
    return 0
