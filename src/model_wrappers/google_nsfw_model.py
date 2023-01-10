import torch

from src.arch.google_nsfw_model import GoogleNSFWModel
from src.model_wrappers.general_model import MeanStdType, ModelWrapper


class GoogleNSFWModelWrapper(ModelWrapper):
    def __init__(self,
                 model: GoogleNSFWModel,
                 n_class=2,
                 im_mean: MeanStdType = None,
                 im_std: MeanStdType = None,
                 take_sigmoid=False,
                 threshold: float = 2.0/5.0):
        self.model = model
        self.threshold = threshold
        super().__init__(n_class, im_mean, im_std, take_sigmoid)

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self._predict_prob(image)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def _predict_prob(self, image: torch.Tensor, verbose=False) -> torch.Tensor:
        prob = self.model.request_classification(image, verbose)
        self.num_queries += image.size(0)
        if self.num_queries % 100 == 0:
            print(f"Ran {self.num_queries} queries on Google NSFW model")
        return prob

    def predict_label_binary(self, image, verbose=False):
        logits = self._predict_prob(image, verbose)

        # 0/5: UNKNOWN
        # 1/5: VERY_UNLIKELY
        # 2/5: UNLIKELY
        # 3/5: POSSIBLE
        # 4/5: LIKELY
        # 5/5: VERY_LIKELY
        predict = torch.gt(logits, self.threshold).to(torch.long)
        return predict

    def predict_label_multiclass(self, image: torch.Tensor, verbose=False) -> torch.Tensor:
        raise NotImplementedError("Multi-class classification not implemented for EdenAI models")
