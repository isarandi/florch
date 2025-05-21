import abc

import torch.nn as nn
from florch.util import attrdict2dict_nested, dict2attrdict_nested
from florch.util.easydict import EasyDict


class ModelTrainer(nn.Module, abc.ABC):
    def __init__(self, model: nn.Module, random_seed: int = 0):
        super().__init__()
        self.model = model
        self.random_seed = random_seed
        self.is_testing = False
        self.grad_accum_steps = 1
        self.train_in_inference_mode = False
        self._train_counter = 0

    def forward(self, inps):
        inps = dict2attrdict_nested(inps)
        preds = self.forward_train(inps)
        losses = self.compute_losses(inps, preds)
        return (
            attrdict2dict_nested(preds),
            attrdict2dict_nested(losses),
        )

    def _forward_test(self, inps):
        inps = dict2attrdict_nested(inps)
        inps = self.prepare_inputs(inps)
        preds = self.forward_test(inps)
        return attrdict2dict_nested(inps), attrdict2dict_nested(preds)

    def _compute_metrics(self, inps, preds):
        inps = dict2attrdict_nested(inps)
        preds = dict2attrdict_nested(preds)
        metrics = self.compute_metrics(inps, preds)
        return attrdict2dict_nested(metrics)

    def _prepare_inputs(self, inps):
        inps = dict2attrdict_nested(inps)
        return attrdict2dict_nested(self.prepare_inputs(inps))

    def prepare_inputs(self, inps):
        return inps

    @abc.abstractmethod
    def forward_train(self, inps):
        pass

    def forward_test(self, inps):
        return self.forward_train(inps)

    @abc.abstractmethod
    def compute_losses(self, inps, preds):
        pass

    def compute_metrics(self, inps, preds):
        return EasyDict()

    @property
    def train_counter(self) -> int:
        return self._train_counter

    @property
    def adjusted_train_counter(self) -> int:
        return self.train_counter // self.grad_accum_steps
