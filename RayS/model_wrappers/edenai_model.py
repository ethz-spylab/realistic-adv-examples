import torch

from arch.edenai_model import EdenAINSFWModel
from model_wrappers.general_model import ModelWrapper


class EdenAIModelWrapper(ModelWrapper):
    def __init__(self, model: EdenAINSFWModel, n_class=10, im_mean=None, im_std=None, take_sigmoid=False):
        self.model = model
        super().__init__(n_class, im_mean, im_std, take_sigmoid)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.predict_prob(image)
        
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image
    
    def predict_prob(self, image: torch.Tensor) -> torch.Tensor:
        prob = self.model.request_classification(image)
        return prob
    
    def predict_label_binary(self, image):
        logits = self.predict_prob(image)
        predict = torch.gt(logits, 0).to(torch.long)
        return predict
    
    def predict_label_multiclass(self, _: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Multi-class classification not implemented for EdenAI models")