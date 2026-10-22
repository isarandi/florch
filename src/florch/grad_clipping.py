"""Adaptive gradient clipping."""

import collections
from typing import Optional

import numpy as np
import torch


class AdaptiveGradientClipper:
    """Adaptively clips gradient norms for multiple parameter groups.

    Instead of using a fixed max-norm, this class keeps a running history of
    recent gradient norms per parameter group. For each group, the clip
    threshold is set to:

        clip_norm = quantile(history) * factor

    so that the threshold tracks the typical gradient scale during training.

    Args:
        quantile: Quantile of the stored history to use as base clipping
            value (e.g. 0.5 -> median, 0.9 -> 90th percentile).
        size: Maximum number of past norms to keep per parameter group.
        factor: Multiplier applied to the quantile-based norm to get the
            final clipping threshold. None = no clipping.
        skip_factor: Norms above skip_factor * clip_norm are treated as outliers:
            the training loop skips such steps, and they are excluded from the
            history so that a sustained spike burst cannot inflate the threshold
            and thereby disarm the clipping.
    """

    def __init__(
        self,
        quantile: float = 0.5,
        size: int = 100,
        factor: Optional[float] = None,
        skip_factor: float = 100.0,
    ):
        self.factor = factor
        self.size = size
        self.quantile = quantile
        self.skip_factor = skip_factor
        self.histories: Optional[list[collections.deque]] = None

    @property
    def enabled(self) -> bool:
        return self.factor is not None

    def clip(self, *param_groups):
        """Clip gradients of the given parameter groups in-place.

        Args:
            *param_groups: Each argument is an iterable of parameters.
                Each group gets its own running history and threshold.

        Returns:
            norms: List of current gradient norms (one per group).
            norm_factors: List of ratios norm / clip_norm.
        """
        if self.histories is None:
            self.histories = [collections.deque() for _ in range(len(param_groups))]

        while len(self.histories) < len(param_groups):
            self.histories.append(collections.deque())

        norms = []
        norm_factors = []
        for hist, param in zip(self.histories, param_groups):
            if len(hist) > 0:
                base = np.quantile(hist, self.quantile)
                clip_norm = base * self.factor
            else:
                clip_norm = np.inf

            norm = torch.nn.utils.clip_grad_norm_(param, max_norm=clip_norm).item()

            # Raw norms enter the history so the adaptive threshold can track genuine
            # regime shifts in gradient scale, including through spikes. Only
            # non-finite norms are excluded - a single NaN/inf would corrupt the
            # quantile permanently.
            if np.isfinite(norm):
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
