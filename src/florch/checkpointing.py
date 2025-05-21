import os
import os.path as osp

import torch


class CheckpointManager:
    def __init__(
        self,
        model,
        optimizer,
        grad_clipper,
        directory,
        max_to_keep,
        checkpoint_interval,
        truncate_before_delete=False,
        fsdp=False,
    ):
        self.fsdp = fsdp
        self.model = model
        self.optimizer = optimizer
        self.grad_clipper = grad_clipper
        self.directory = directory
        self.max_to_keep = max_to_keep
        os.makedirs(directory, exist_ok=True)
        self.checkpoints = sorted(self._get_existing_checkpoints())
        self.checkpoint_interval = checkpoint_interval
        self.truncate_before_delete = truncate_before_delete

    def _get_existing_checkpoints(self):
        """Retrieve existing checkpoint paths sorted by modification time."""
        return sorted(
            [
                osp.join(self.directory, ckpt)
                for ckpt in os.listdir(self.directory)
                if ckpt.endswith(".pth")
            ],
            key=osp.getmtime,
        )

    def save(self, step, name=None, check_interval=True, overwrite=False):
        """Save a new checkpoint and remove old ones if exceeding max_to_keep."""

        if check_interval and step % self.checkpoint_interval != 0:
            return

        if name is None:
            name = f"ckpt-{step:07d}"

        checkpoint_path = osp.join(self.directory, f"{name}.pth")
        if not overwrite and osp.exists(checkpoint_path):
            return

        if self.fsdp:
            from torch.distributed.checkpoint.state_dict import get_state_dict

            model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        else:
            model_state_dict = self.model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()

        torch.save(
            {
                'n_completed_steps': step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'grad_clipper_state_dict': self.grad_clipper.state_dict(),
            },
            checkpoint_path,
        )

        self.checkpoints.append(checkpoint_path)
        self.remove_old_checkpoints()

    def remove_old_checkpoints(self):
        while len(self.checkpoints) > self.max_to_keep:
            path_to_delete = self.checkpoints.pop(0)
            if self.truncate_before_delete:
                # Truncate the checkpoint file before deleting it
                with open(path_to_delete, 'wb') as f:
                    f.truncate()

            os.remove(path_to_delete)

    def load_latest(self):
        """Load the latest checkpoint if available."""
        if not self.checkpoints:
            return None, 0  # No checkpoint found
        latest_checkpoint = self.checkpoints[-1]
        checkpoint = torch.load(latest_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.grad_clipper.load_state_dict(checkpoint['grad_clipper_state_dict'])
        return latest_checkpoint, checkpoint['n_completed_steps']


def latest_checkpoint(directory):
    """Retrieve the latest checkpoint in a directory."""
    checkpoints = sorted(
        [
            osp.join(directory, ckpt)
            for ckpt in os.listdir(directory)
            if osp.basename(ckpt).startswith('ckpt-') and ckpt.endswith(".pth")
        ],
    )
    return checkpoints[-1] if checkpoints else None
