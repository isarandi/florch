"""Florch: Training framework with precise resume and stream-based data loading."""

from florch.config import TrainingConfig
from florch.training.job import TrainingJob, DataLoaders
from florch.model_trainer import ModelTrainer
from florch.checkpointing import CheckpointManager
from florch.grad_clipping import AdaptiveGradientClipper
from florch.streams import DataStream, RoundRobinStream, MultiStream, stream_to_loader
from florch.exceptions import UnexpectedValueDuringTrainingError
from florch import callbacks

__all__ = [
    'TrainingConfig',
    'TrainingJob',
    'DataLoaders',
    'ModelTrainer',
    'CheckpointManager',
    'AdaptiveGradientClipper',
    'DataStream',
    'RoundRobinStream',
    'MultiStream',
    'stream_to_loader',
    'BarecatReader',
    'UnexpectedValueDuringTrainingError',
    'callbacks',
]


def __getattr__(name):
    # BarecatReader needs the optional barecat package; import it only on use.
    if name == 'BarecatReader':
        from florch.barecat_reader import BarecatReader
        return BarecatReader
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
