import abc

import torch
from torch import nn

MeanStdType = tuple[float, float, float] | None


class ModelWrapper(nn.Module):

    def __init__(self,
                 n_class: int = 10,
                 im_mean: MeanStdType = None,
                 im_std: MeanStdType = None,
                 take_sigmoid: bool = True):
        super().__init__()
        self.num_queries = 0
        self.im_mean = torch.Tensor(im_mean).view(1, 3, 1, 1).cuda() if im_mean is not None else None
        self.im_std = torch.Tensor(im_std).view(1, 3, 1, 1).cuda() if im_std is not None else None
        self.n_class = n_class
        self.take_sigmoid = take_sigmoid
        if self.n_class == 2:
            print("Using binary predict label function")
            self.predict_label = self.predict_label_binary
        else:
            self.predict_label = self.predict_label_multiclass

    def make_model_eval(self):
        pass

    @abc.abstractmethod
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def _predict_prob(self, image: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        ...

    def predict_label_multiclass(self, image: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        logits = self._predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict

    def predict_label_binary(self, image: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        logits = self._predict_prob(image)
        if self.take_sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        predict = torch.round(probs).to(torch.long)
        return predict
