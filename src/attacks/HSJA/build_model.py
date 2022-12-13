from __future__ import absolute_import, division, print_function

import abc
import math

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from keras import backend as K
from keras import optimizers
from torch import nn


def construct_original_network(dataset_name, model_name, train):
    data_model = dataset_name + model_name

    # Define the model
    input_size = 32
    num_classes = 10
    channel = 3

    assert model_name == 'resnet'
    from resnet import resnet_v2

    model, image_ph, preds = resnet_v2(input_shape=(input_size, input_size, channel), depth=20, num_classes=num_classes)

    optimizer = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    grads = []
    for c in range(num_classes):
        grads.append(tf.gradients(preds[:, c], image_ph))

    grads = tf.concat(grads, axis=0)
    approxs = grads * tf.expand_dims(image_ph, 0)

    logits = [layer.output for layer in model.layers][-2]
    print(logits)

    sess = K.get_session()

    return image_ph, preds, grads, approxs, sess, model, num_classes, logits


class ImageModel(abc.ABC):
    binary: bool
    binary_threshold: float = 0.5

    @abc.abstractmethod
    def predict(self, x: np.ndarray, verbose: int = 0, batch_size: int = 500, logits: bool = False) -> np.ndarray:
        ...

    def predict_label(self, x: np.ndarray, verbose: int = 0, batch_size: int = 500, logits: bool = False) -> np.ndarray:
        out = self.predict(x, verbose, batch_size, logits)
        if self.binary:
            return np.greater_equal(out, self.binary_threshold)
        return np.argmax(out, axis=1)


def expand_input_to_batch(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 3:
        return np.expand_dims(x, 0)
    return x


def expand_pred_to_batch(pred: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 3:
        return pred.reshape(-1)
    return pred


class KerasImageModel(ImageModel):
    binary = False

    def __init__(self, model_name, dataset_name, train=False, load=False, **kwargs):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.data_model = dataset_name + model_name
        self.framework = 'keras'

        print('Constructing network...')
        self.input_ph, self.preds, self.grads, self.approxs, self.sess, self.model, self.num_classes, self.logits = construct_original_network(
            self.dataset_name, self.model_name, train=train)

        self.layers = self.model.layers
        self.last_hidden_layer = self.model.layers[-3]

        self.y_ph = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        if load:
            if load is True:
                print('Loading model weights...')
                self.model.load_weights('{}/models/original.hdf5'.format(self.data_model), by_name=True)
            elif load is not False:
                self.model.load_weights('{}/models/{}.hdf5'.format(self.data_model, load), by_name=True)

    def predict(self, x, verbose=0, batch_size=500, logits=False):
        _x = expand_input_to_batch(x)

        if not logits:
            prob = self.model.predict(_x, batch_size=batch_size, verbose=verbose)
        else:
            num_iters = int(math.ceil(len(_x) * 1.0 / batch_size))
            probs = []
            for i in range(num_iters):
                x_batch = _x[i * batch_size:(i + 1) * batch_size]

                prob = self.sess.run(self.logits, feed_dict={self.input_ph: x_batch})

                probs.append(prob)

            prob = np.concatenate(probs, axis=0)

        return expand_pred_to_batch(prob, x)


class KerasV2ImageModel(ImageModel):
    def __init__(self, model: tf.keras.Model, binary: bool, mean: tuple[float, float, float] | None,
                 std: tuple[float, float, float] | None) -> None:
        super().__init__()
        self.binary = binary
        self.model = model
        self.mean = np.array(mean).reshape(1, 3, 1, 1)
        self.std = np.array(std).reshape(1, 3, 1, 1)

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def predict(self, x: np.ndarray, verbose: int = 0, batch_size: int = 500, logits: bool = False) -> np.ndarray:
        if logits:
            raise ValueError("Logits mode not implemented for `KerasV2ImageModel`.")
        out = self.model.predict(x, batch_size=batch_size, verbose=verbose)
        return out


class TorchImageModel(ImageModel):
    def __init__(self, model: nn.Module, device: torch.device, mean: tuple[float, float, float] | None,
                 std: tuple[float, float, float] | None) -> None:
        super().__init__()
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.mean = torch.tensor(mean).to(self.device).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std).to(self.device).reshape(1, 3, 1, 1)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def predict(self, x: np.ndarray, verbose: int = 0, batch_size: int = 500, logits: bool = False) -> np.ndarray:
        _x = expand_input_to_batch(x)
        x_tensor = torch.from_numpy(_x).float().to(self.device)
        preprocessed_x = self.preprocess(x_tensor)
        out_logits = self.model(preprocessed_x)
        if logits:
            out = out_logits.cpu().numpy()
        else:
            out = F.softmax(out_logits).cpu().numpy()

        return expand_pred_to_batch(out, x)
