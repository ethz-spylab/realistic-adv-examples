import torch
import tensorflow as tf
from general_torch_model import GeneralTorchModel


class GeneralTFModel(GeneralTorchModel):
    
    def make_model_eval(self):
        pass

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image_tf = tf.constant(image.cpu())
        logits = self.model(image_tf).numpy()
        return torch.from_numpy(logits).cuda()

    def predict_prob(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        image_tf = tf.constant(image.cpu())
        logits = self.model(image_tf).numpy()
        self.num_queries += image.size(0)
        return torch.from_numpy(logits).cuda()
