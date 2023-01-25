from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models


class BinaryResNet50(LightningModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        fc_in_size = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=fc_in_size, out_features=1)
        self.val_accuracy = torchmetrics.Accuracy(num_classes=1)
        self.val_precision = torchmetrics.Precision(num_classes=1)
        self.val_recall = torchmetrics.Recall(num_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(  # type: ignore
            self, train_batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = train_batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y)
        accuracy = (torch.round(torch.sigmoid(y_hat)) == y).to(y.dtype).mean()
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', accuracy, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):  # type: ignore
        x, y = val_batch
        y_hat = self(x).flatten()
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(x.dtype))
        self.val_accuracy.update(y_hat.sigmoid(), y)
        self.val_precision.update(y_hat.sigmoid(), y)
        self.val_recall.update(y_hat.sigmoid(), y)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', self.val_accuracy, sync_dist=True, prog_bar=True)
        self.log('val_precision', self.val_precision, sync_dist=True, prog_bar=True)
        self.log('val_recall', self.val_recall, sync_dist=True, prog_bar=True)
