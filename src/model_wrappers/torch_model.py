import torch
import torch.nn as nn

from src.model_wrappers.general_model import ModelWrapper, MeanStdType


class TorchModelWrapper(ModelWrapper):

    def __init__(self,
                 model: nn.Module,
                 n_class: int = 10,
                 im_mean: MeanStdType = None,
                 im_std: MeanStdType = None,
                 take_sigmoid: bool = True):
        self._model = model
        super().__init__(n_class, im_mean, im_std, take_sigmoid)

    def make_model_eval(self):
        self._model.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self._model(image)
        return logits

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        processed = image
        if self.im_mean is not None and self.im_std is not None:
            processed = (image - self.im_mean) / self.im_std
        return processed

    def _predict_prob(self, image: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            logits = self._model(image)
            self.num_queries += image.size(0)
        return logits
