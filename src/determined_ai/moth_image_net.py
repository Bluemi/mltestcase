from typing import Union, Dict, Any

import torch
from torch import optim, nn, Tensor
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, TorchData, DataLoader
from torchvision import transforms

from model.layers import Conv2dMoth
from model.resnet import ResNet18
from utils.datasets import get_data_dir, ImageNetDataset


class MothTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext):
        super().__init__(context)
        self.context = context

        self.model = self.context.wrap_model(self._build_model())

        # optimizer
        self.optimizer = self.context.wrap_optimizer(self._build_optimizer())

        # loss function
        self.loss_function = nn.MSELoss()

    def _build_model(self):
        activation_func = self.context.get_hparam('activation_func')
        match activation_func:
            case 'normal':
                layer_type = nn.Conv2d
            case 'moth':
                layer_type = Conv2dMoth
            case _:
                raise ValueError(f'Unknown activation function: {activation_func}')

        return ResNet18(layer_type=layer_type)

    def _build_optimizer(self):
        return optim.SGD(
            self.model.parameters(),
            lr=self.context.get_hparam('lr'),
            weight_decay=self.context.get_hparam('weight_decay'),
            momentum=self.context.get_hparam('momentum')
        )

    def train_batch(
            self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Union[Tensor, Dict[str, Any]]:
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

    def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, Any]:
        inputs, labels = batch
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, torch.flatten(inputs, start_dim=1))

        # TODO: write images to determined ai

        return {
            'loss': loss
        }

    def build_training_data_loader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Normalize(ImageNetDataset.MEAN_VALUES, ImageNetDataset.STD_VALUES)
        ])

        datadir = get_data_dir() / 'train'

        dataset = ImageNetDataset(root=datadir, transform=transform)

        return DataLoader(
            dataset,
            batch_size=self.context.get_global_batch_size(),
            shuffle=True,
            num_workers=0,
        )

    def build_validation_data_loader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Normalize(ImageNetDataset.MEAN_VALUES, ImageNetDataset.STD_VALUES)
        ])

        datadir = get_data_dir() / 'validation'

        dataset = ImageNetDataset(root=datadir, transform=transform)

        return DataLoader(
            dataset,
            batch_size=self.context.get_global_batch_size(),
            shuffle=True,
            num_workers=0,
        )
