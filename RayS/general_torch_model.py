import torch
import numpy as np
import torch.nn as nn


class GeneralTorchModel(nn.Module):

    def __init__(self, model, n_class=10, im_mean=None, im_std=None, take_sigmoid=True):
        super().__init__()
        self.model = model
        self.make_model_eval()
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
        self.model.eval()

    def forward(self, image):
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self.model(image)
        return logits

    def preprocess(self, image):
        if isinstance(image, np.ndarray):
            processed = torch.from_numpy(image).to(torch.float)
        else:
            processed = image

        if self.im_mean is not None and self.im_std is not None:
            processed = (processed - self.im_mean) / self.im_std
        return processed

    def predict_prob(self, image):
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            logits = self.model(image)
            self.num_queries += image.size(0)
        return logits

    def predict_label_multiclass(self, image):
        logits = self.predict_prob(image)
        _, predict = torch.max(logits, 1)
        return predict

    def predict_label_binary(self, image):
        logits = self.predict_prob(image)
        if self.take_sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits
        predict = torch.round(probs).to(torch.long)
        return predict
