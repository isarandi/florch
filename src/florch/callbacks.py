from typing import Optional, Sequence, TYPE_CHECKING, Union

import florch.checkpointing
import florch.layers.batch_renorm
import simplepyutils as spu
import torch.distributed
import torch.nn as nn
import numpy as np

if TYPE_CHECKING:
    from florch import ModelTrainer


class Callback:
    def __init__(self):
        self.trainer: Optional['ModelTrainer'] = None
        self.device = None

    def on_train_batch_begin(self, step: int):
        pass

    def on_train_batch_end(self, step: int, logs: dict[str, float]):
        pass

    def on_train_begin(self, initial_step: int):
        pass

    def on_train_end(self, step: int):
        pass

    def needs_to_run_on_all_devices(self):
        return False


class Checkpoint(Callback):
    def __init__(self, ckpt_manager: florch.checkpointing.CheckpointManager):
        super().__init__()
        self.ckpt_manager = ckpt_manager

    def on_train_batch_end(self, step, logs):
        self.ckpt_manager.save(step + 1, check_interval=True)


class Progbar(Callback):
    def __init__(self, total):
        super().__init__()
        self.total = total

    def on_train_begin(self, initial_step):
        self.pbar = spu.progressbar(initial=initial_step, total=self.total)

    def on_train_batch_end(self, step, logs):
        self.pbar.n = step
        self.pbar.update(0)
        self.pbar.set_postfix(logs)


class Wandb(Callback):
    def __init__(self, logdir, config_dict, project_name, every_n_steps=30):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.logdir = logdir
        self.config_dict = config_dict
        self.project_name = project_name
        self.accum_metrics = {}
        self.n_accumulated = 0

    def on_train_begin(self, initial_step):
        import wandb.util

        id_path = f"{self.logdir}/run_id"
        try:
            with open(id_path) as f:
                run_id = f.read()
        except FileNotFoundError:
            run_id = wandb.util.generate_id()
            with open(id_path, "w") as f:
                f.write(str(run_id))
                f.flush()

        wandb.init(
            name=self.logdir.split("/")[-1],
            project=self.project_name,
            config=self.config_dict,
            dir=self.logdir,
            id=run_id,
            resume="allow",
            settings=wandb.Settings(_service_wait=300, init_timeout=300),
        )

    def on_train_batch_end(self, step, logs):
        import wandb

        for key, value in logs.items():
            if key not in self.accum_metrics:
                self.accum_metrics[key] = 0.0
            self.accum_metrics[key] += value
        self.n_accumulated += 1

        val_metrics = {k: v for k, v in logs.items() if k.startswith("val_")}

        if step % self.every_n_steps == 0:
            logs = {k: v / self.n_accumulated for k, v in self.accum_metrics.items()} | val_metrics
            self.accum_metrics.clear()
            self.n_accumulated = 0
        else:
            logs = val_metrics

        if logs:
            wandb.log(logs, step=int(step), commit=True)


class SyncBNStats(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, step, logs):
        self.trainer.model.apply(self.sync_running_stats)

    def needs_to_run_on_all_devices(self):
        return True

    @staticmethod
    def has_any_bn_layers(module):
        for m in module.modules():
            if isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.LazyBatchNorm1d,
                    nn.LazyBatchNorm2d,
                    nn.LazyBatchNorm3d,
                    florch.layers.batch_renorm.BatchRenorm1d,
                    florch.layers.batch_renorm.BatchRenorm2d,
                    florch.layers.batch_renorm.BatchRenorm3d,
                ),
            ):
                return True
        return False

    @staticmethod
    def sync_running_stats(module):
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                florch.layers.batch_renorm.BatchRenorm1d,
                florch.layers.batch_renorm.BatchRenorm2d,
                florch.layers.batch_renorm.BatchRenorm3d,
            ),
        ):
            torch.distributed.all_reduce(module.running_mean, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(module.running_var, op=torch.distributed.ReduceOp.AVG)


class SwitchToInferenceModeCallback(Callback):
    def __init__(self, step_to_switch_to_inference_mode, ckpt_manager):
        super().__init__()
        self.step_to_switch_to_inference_mode = step_to_switch_to_inference_mode
        self.ckpt_manager = ckpt_manager

    def on_train_batch_end(self, batch, logs):
        if (
            batch >= self.step_to_switch_to_inference_mode
            and not self.trainer.train_in_inference_mode
        ):
            self.ckpt_manager.save(
                batch + 1, name="ckpt_before_switch_to_inference_mode", check_interval=False
            )
            self.trainer.train_in_inference_mode = True

    def needs_to_run_on_all_devices(self):
        return True


class ConvMinMaxNormConstraint(Callback):
    def __init__(
        self, rate: float = 1.0, min_value: float = 0.0, max_value: float = 1.0, eps: float = 1e-7
    ):
        super().__init__()
        self.rate = rate
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps

    def on_train_batch_end(self, step, logs):
        with torch.no_grad():
            self.trainer.model.apply(self.apply_constraint)

    def apply_constraint(self, module: nn.Module):
        if isinstance(module, nn.Conv2d) and module.weight is not None:
            if module.groups == module.in_channels:  # Depthwise Conv2D
                module.weight.copy_(
                    self.minmax_norm(module.weight, dim=[-1, -2])
                )  # Norm over h, w
            else:  # Standard Conv2D
                module.weight.copy_(
                    self.minmax_norm(module.weight, dim=[-1, -2, -3])
                )  # Norm over c_in, h, w

    def minmax_norm(self, w: torch.Tensor, dim: Union[int, Sequence[int]]):
        norms = torch.sqrt(torch.sum(torch.square(w), dim=dim, keepdim=True))  # Compute L2 norm
        desired = (
            self.rate * torch.clamp(norms, self.min_value, self.max_value)
            + (1.0 - self.rate) * norms
        )
        return w * (desired / (norms + self.eps))

    def needs_to_run_on_all_devices(self):
        return True


class MinMaxNormConstraint(Callback):
    def __init__(
        self,
        parameters: Sequence[nn.Parameter],
        rate: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        dim: Optional[Union[int, Sequence[int]]] = None,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.parameters = parameters
        self.rate = rate
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps
        self.dim = dim

    def on_train_batch_end(self, step, logs):
        with torch.no_grad():
            for p in self.parameters:
                p.copy_(self.minmax_norm(p, dim=self.dim))

    def minmax_norm(self, w: torch.Tensor, dim: Union[int, Sequence[int]]):
        norms = torch.sqrt(torch.sum(torch.square(w), dim=dim, keepdim=True))  # Compute L2 norm
        desired = (
            self.rate * torch.clamp(norms, self.min_value, self.max_value)
            + (1.0 - self.rate) * norms
        )
        return w * (desired / (norms + self.eps))

    def needs_to_run_on_all_devices(self):
        return True


class FreezeLayers(Callback):
    def __init__(self, trainables: Sequence[Union[nn.Module, nn.Parameter]], unfreeze_step: int):
        super().__init__()
        self.unfreeze_step = unfreeze_step
        self.trainables = trainables

    def on_train_begin(self, initial_step: int):
        requires_grad = initial_step >= self.unfreeze_step
        for t in self.trainables:
            t.requires_grad_(requires_grad)

    def on_train_batch_begin(self, step, logs=None):
        if step == self.unfreeze_step:
            for t in self.trainables:
                t.requires_grad_(True)

    def needs_to_run_on_all_devices(self):
        return True


class TransitionBatchNorm(Callback):
    def __init__(
        self,
        transition_start: int,
        transition_end: int,
    ):
        super().__init__()
        self.transition_start = transition_start
        self.transition_end = transition_end

    def on_train_batch_begin(self, step: int):
        blend = self.smootherstep(
            step, self.transition_start, self.transition_end
        )
        for m in self.trainer.model.modules():
            if isinstance(m, florch.layers.TransitionBatchNorm2d):
                m.blend = blend

        if step == self.transition_start:
            for m in self.trainer.model.modules():
                if isinstance(m, florch.layers.TransitionBatchNorm2d):
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)


    @staticmethod
    def smootherstep(x, x0=0, x1=None, length=None):
        if length is None:
            length = x1 - x0
        y = np.clip((x - x0) / length, 0, 1)
        return y * y * y * (y * (y * 6.0 - 15.0) + 10.0)


    def needs_to_run_on_all_devices(self):
        return True
