from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from pytorch_lightning import LightningModule

class BinaryResNet50(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        fc_in_size = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=fc_in_size, out_features=1)
        
    def forward(self, x) -> Any:
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y)
        accuracy = (torch.round(torch.sigmoid(y_hat)) == y).to(y.dtype).mean()
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', accuracy, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y)
        accuracy = (torch.round(torch.sigmoid(y_hat)) == y).to(y.dtype).mean()
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_acc', accuracy, sync_dist=True, prog_bar=True)