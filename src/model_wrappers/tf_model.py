import tensorflow as tf
import torch

from src.model_wrappers.general_model import MeanStdType, ModelWrapper


class TFModelWrapper(ModelWrapper):
    def __init__(self,
                 model: tf.keras.Model,
                 n_class: int = 10,
                 im_mean: MeanStdType = None,
                 im_std: MeanStdType = None,
                 take_sigmoid: bool = True):
        self._model = model
        super().__init__(n_class, im_mean, im_std, take_sigmoid)

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        processed = image
        if self.im_mean is not None and self.im_std is not None:
            processed = (image - self.im_mean) / self.im_std
        return processed

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image_tf = tf.constant(image.cpu())
        logits = self._model(image_tf).numpy()
        return torch.from_numpy(logits).cuda()

    def _predict_prob(self, image: torch.Tensor, verbose=False) -> torch.Tensor:
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        image_tf = tf.constant(image.cpu())
        logits = self._model(image_tf).numpy()
        self.num_queries += image.size(0)
        return torch.from_numpy(logits).cuda()
