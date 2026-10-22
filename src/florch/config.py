"""Training configuration dataclass."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TrainingConfig:
    """Configuration for TrainingJob.

    Groups all training hyperparameters into a single dataclass for:
    - Cleaner __init__ signatures
    - Easy serialization/loading
    - IDE autocompletion

    Example:
        config = TrainingConfig(
            training_steps=100000,
            grad_accum_steps=4,
            checkpoint_dir='checkpoints/',
            wandb_project='my-project',
        )
        job = MyTrainingJob(config)
        job.train()
    """

    # === Core training ===
    training_steps: int
    """Total number of training steps (optimizer updates)."""

    seed: int = 0
    """Random seed for reproducibility."""

    # === Batching ===
    grad_accum_steps: int = 1
    """Number of gradient accumulation steps before optimizer update."""

    # === Checkpointing ===
    checkpoint_dir: Optional[str] = None
    """Directory for saving checkpoints."""

    checkpoint_period: int = 1000
    """Save checkpoint every N steps."""

    checkpoint_max_to_keep: int = 2
    """Number of most recent checkpoints to keep (older ones are pruned)."""

    # === Resume ===
    load_path: Optional[str] = None
    """Path to checkpoint to resume from."""

    init_path: Optional[str] = None
    """Path to checkpoint for weight initialization only (not optimizer)."""

    n_completed_steps: Optional[int] = None
    """Override detected completed steps (usually auto-detected from checkpoint)."""

    stop_step: Optional[int] = None
    """Stop training at this step (default: training_steps)."""

    # === Distributed ===
    multi_gpu: bool = False
    """Enable multi-GPU training with DDP."""

    find_unused_parameters: bool = True
    """DDP find_unused_parameters flag. Set True when different substeps use
    different model parts (e.g. multi-stream training)."""

    use_fsdp: bool = False
    """Use FSDP instead of DDP for large models."""

    fsdp_sharding_strategy: str = "FULL_SHARD"
    """FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD."""

    fsdp_cpu_offload: bool = False
    """Offload parameters and gradients to CPU."""

    fsdp_mixed_precision: Optional[str] = None
    """FSDP mixed precision: 'fp16', 'bf16', or None for fp32."""

    fsdp_backward_prefetch: str = "BACKWARD_PRE"
    """FSDP backward prefetch: BACKWARD_PRE, BACKWARD_POST."""

    fsdp_use_orig_params: bool = True
    """Use original params for torch.compile compatibility."""

    # === Mixed precision ===
    loss_scale: float = 1.0
    """Initial loss scale for mixed precision. Set > 1 for float16."""

    # === Gradient clipping ===
    clip_grad_norm_factor: Optional[float] = None
    """Adaptive gradient clipping factor. None = no clipping.
    Clips to quantile(history) * factor."""

    clip_grad_norm_quantile: float = 0.5
    """Quantile of gradient norm history for adaptive clipping."""

    clip_grad_norm_histsize: int = 100
    """Size of gradient norm history for adaptive clipping."""

    clip_grad_norm_skip_factor: float = 100.0
    """Steps with gradient norm above skip_factor * clip threshold are skipped
    entirely and excluded from the norm history."""

    # === Regularization ===
    norm_loss_factor: float = 0.0
    """Weight for norm layer regularization loss."""

    # NOTE: norm_loss_ramp_steps is the absolute step at which the smootherstep ramp
    # ENDS (x1), not a duration - it must be larger than norm_loss_start_step.

    norm_loss_start_step: int = 1000
    """Step to start norm loss regularization."""

    norm_loss_ramp_steps: int = 4000
    """Steps to ramp up norm loss from 0 to full weight."""

    # === Validation ===
    validate_period: int = 0
    """Run validation every N steps. 0 = no validation."""

    # === Logging ===
    wandb_project: Optional[str] = None
    """Weights & Biases project name. None = no wandb logging."""

    wandb_config: Any = None
    """Config dict to log to wandb."""

    logdir: Optional[str] = None
    """Directory for logs and artifacts."""

    # === Compilation ===
    compile: bool = True
    """Whether to use torch.compile on the trainer. Set False for debugging."""

    # === Advanced ===
    finetune_in_inference_mode: int = 0
    """Switch to inference mode for last N steps (BatchNorm in eval mode)."""

    workers: Optional[int] = None
    """Number of data loading workers. None = auto."""

    parallel_build_data: bool = True
    """Build data pipeline in parallel with model building."""

    def __post_init__(self):
        if self.stop_step is None:
            self.stop_step = self.training_steps

        if self.checkpoint_dir is None and self.logdir is not None:
            self.checkpoint_dir = f"{self.logdir}/checkpoints"
