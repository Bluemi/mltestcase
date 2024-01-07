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

        self.model = self.context.wrap_model(self._build_model())

        # optimizer
        # TODO: learning rate scheduler
        self.optimizer = self.context.wrap_optimizer(self._build_optimizer())

        # loss function
        self.loss_function = nn.MSELoss()

    def _build_model(self):
        activation_func = self.context.get_hparam('activation_func')
        use_activation_for_z = self.context.get_hparam('use_activation_for_z')
        return MnistAutoencoder(activation_func=activation_func, use_activation_for_z=use_activation_for_z)

    def _build_optimizer(self):
        lr = self.context.get_hparam('lr')
        weight_decay = self.context.get_hparam('weight_decay')
        return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

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
