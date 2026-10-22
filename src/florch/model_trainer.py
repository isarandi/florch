"""Base class for model trainers."""

import abc

import torch
import torch.nn as nn


class ModelTrainer(nn.Module, abc.ABC):
    """nn.Module wrapping a model with forward pass and loss computation.

    This is the unit that gets torch.compile'd and DDP/FSDP-wrapped.
    Keep it thin: just the model, forward pass, and loss computation.

    Other training concerns (gradient clipping param groups) belong on TrainingJob.

    Example:
        class MyTrainer(ModelTrainer):
            def forward_train(self, inps):
                logits = self.model(inps['image'])
                return {'logits': logits}

            def compute_losses(self, inps, preds):
                loss = F.cross_entropy(preds['logits'], inps['label'])
                return {'loss': loss}
    """

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self._train_counter = torch.tensor(0, dtype=torch.int64)
        self.grad_accum_steps = 1
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def adjusted_train_counter(self) -> int:
        return self._train_counter // self.grad_accum_steps

    def set_training_state(self, substep, grad_accum_steps):
        """Set training loop state (called by TrainingJob before the loop)."""
        self._train_counter.fill_(substep)
        self.grad_accum_steps = grad_accum_steps

    def forward(self, inps):
        """Forward pass for DDP/compile wrapping. Returns (preds, losses)."""
        preds = self.forward_train(inps)
        losses = self.compute_losses(inps, preds)
        return preds, losses

    def _prepare_inputs(self, inps):
        return self.prepare_inputs(inps)

    def _compute_metrics(self, inps, preds):
        return self.compute_metrics(inps, preds)

    def prepare_inputs(self, inps):
        """Preprocess batch before forward pass. Called outside DDP/compile wrapper."""
        return inps

    @abc.abstractmethod
    def forward_train(self, inps):
        """Forward pass during training. Returns predictions dict."""
        pass

    @abc.abstractmethod
    def compute_losses(self, inps, preds):
        """Compute training losses. Must return dict with 'loss' key."""
        pass

    def compute_metrics(self, inps, preds):
        """Compute metrics for logging. Called with torch.inference_mode()."""
        return {}
