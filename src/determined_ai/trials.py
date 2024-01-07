from typing import Union, Dict, Any

import torch
from determined import pytorch

from determined.pytorch import PyTorchTrial, PyTorchTrialContext
from torch import optim, nn

from model import MnistAutoencoder
from utils.datasets import load_data


class AutoencoderTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        super().__init__(context)
        self.context = context

        self.model = self.context.wrap_model(MnistAutoencoder())

        # optimizer
        # TODO: learning rate scheduler
        self.optimizer = self.context.wrap_optimizer(
            optim.AdamW(self.model.parameters(), lr=0.002, weight_decay=0.0002)
        )

        # loss function
        self.loss_function = nn.MSELoss()

    def train_batch(
        self, batch: pytorch.TorchData, epoch_idx: int, batch_idx: int
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        inputs, labels = batch
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.loss_function(outputs, torch.flatten(inputs, start_dim=1))
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        return {
            'loss': loss,
            # 'scheduler_learning_rate': self.scheduler.get_last_lr()[0],
        }

    def evaluate_batch(self, batch: pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        inputs, labels = batch
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, torch.flatten(inputs, start_dim=1))

        # TODO: write images to determined ai

        return {
            'loss': loss
        }

    def build_training_data_loader(self) -> pytorch.DataLoader:
        return load_data(
            'mnist',
            train=True,
            batch_size=self.context.get_global_batch_size(),
            num_workers=0,
            device=self.context.device
        )

    def build_validation_data_loader(self) -> pytorch.DataLoader:
        return load_data(
            'mnist',
            train=False,
            batch_size=self.context.get_global_batch_size(),
            num_workers=0,
            device=self.context.device
        )
