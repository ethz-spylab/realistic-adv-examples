import torch

from arch.edenai_model import EdenAINSFWModel
from model_wrappers.general_model import ModelWrapper


class EdenAIModelWrapper(ModelWrapper):
    def __init__(self, model: EdenAINSFWModel, n_class=10, im_mean=None, im_std=None, take_sigmoid=False, threshold: float = 0):
        self.model = model
        self.threshold = threshold
        super().__init__(n_class, im_mean, im_std, take_sigmoid)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.predict_prob(image)
        
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image
    
    def predict_prob(self, image: torch.Tensor, verbose=False) -> torch.Tensor:
        prob = self.model.request_classification(image, verbose)
        return prob
    
    def predict_label_binary(self, image, verbose=False):
        logits = self.predict_prob(image, verbose)
        predict = torch.gt(logits, self.threshold).to(torch.long)
        return predict
    
    def predict_label_multiclass(self, _: torch.Tensor, verbose=False) -> torch.Tensor:
        raise NotImplementedError("Multi-class classification not implemented for EdenAI models")