from typing import Union, Dict, Any

import torch
from torch import optim, nn, Tensor
from determined.pytorch import PyTorchTrial, PyTorchTrialContext, TorchData, DataLoader
from torchvision import transforms
import torchmetrics

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
        self.loss_function = nn.CrossEntropyLoss()

        self.top5_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=1000, top_k=5)
        self.top1_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=1000, top_k=1)

    def _build_model(self):
        activation_func = self.context.get_hparam('activation_func')
        match activation_func:
            case 'normal':
                layer_type = nn.Conv2d
            case 'moth':
                layer_type = Conv2dMoth
            case _:
                raise ValueError(f'Unknown activation function: {activation_func}')

        return ResNet18(layer_type=layer_type, num_classes=1000)

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
        loss = self.loss_function(outputs, labels)
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
            loss = self.loss_function(outputs, labels)

            top1_accuracy = self.top1_accuracy(outputs, labels)
            top5_accuracy = self.top5_accuracy(outputs, labels)

        # TODO: write images to determined ai

        return {
            'loss': loss,
            'accuracy_top1': top1_accuracy,
            'accuracy_top5': top5_accuracy,
        }

    def build_training_data_loader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Normalize(ImageNetDataset.MEAN_VALUES, ImageNetDataset.STD_VALUES)
        ])

        datadir = get_data_dir() / 'datasets' / 'ImageNet'

        dataset = ImageNetDataset(root=datadir, transform=transform, train=True)

        return DataLoader(
            dataset,
            batch_size=self.context.get_global_batch_size(),
            shuffle=True,
            num_workers=8,
        )

    def build_validation_data_loader(self) -> DataLoader:
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Normalize(ImageNetDataset.MEAN_VALUES, ImageNetDataset.STD_VALUES)
        ])

        datadir = get_data_dir() / 'datasets' / 'ImageNet'

        dataset = ImageNetDataset(root=datadir, transform=transform, train=False)

        return DataLoader(
            dataset,
            batch_size=self.context.get_global_batch_size(),
            shuffle=False,
            num_workers=8,
        )
