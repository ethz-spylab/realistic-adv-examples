import sys
from argparse import ArgumentParser
from pathlib import Path

import autokeras as ak
import torch
import tensorflow as tf

sys.path.append(str(Path(__file__).parent.parent))
from src.arch.clip_laion_nsfw import PortedCLIPClassifier, BASE_CHECKPOINT_PATH


def port_keras_model(checkpoints_dir: str = "checkpoints") -> PortedCLIPClassifier:
    ported_classifier = PortedCLIPClassifier(512)
    keras_model: tf.keras.Model = tf.keras.models.load_model(
        BASE_CHECKPOINT_PATH.format(  # type: ignore
            checkpoints_dir=checkpoints_dir, model="b32"),
        custom_objects=ak.CUSTOM_OBJECTS)
    encoding_layers = keras_model.get_layer("multi_category_encoding").encoding_layers  # type: ignore
    category_mapper = torch.tensor([[1. if layer is None else 0. for layer in encoding_layers]])
    ported_classifier.category_mapper = category_mapper
    ported_classifier.mean = torch.from_numpy(keras_model.get_layer("normalization").mean.numpy())  # type: ignore
    ported_classifier.std = torch.sqrt(torch.from_numpy(
        keras_model.get_layer("normalization").variance.numpy()))  # type: ignore
    ported_classifier.linear_1.weight = torch.nn.Parameter(
        torch.from_numpy(keras_model.get_layer("dense").kernel.numpy()).t())  # type: ignore
    ported_classifier.linear_1.bias = torch.nn.Parameter(torch.from_numpy(
        keras_model.get_layer("dense").bias.numpy()))  # type: ignore
    ported_classifier.linear_2.weight = torch.nn.Parameter(
        torch.from_numpy(keras_model.get_layer("dense_1").kernel.numpy()).t())  # type: ignore
    ported_classifier.linear_2.bias = torch.nn.Parameter(torch.from_numpy(
        keras_model.get_layer("dense_1").bias.numpy()))  # type: ignore
    ported_classifier.linear_3.weight = torch.nn.Parameter(
        torch.from_numpy(keras_model.get_layer("dense_2").kernel.numpy()).t())  # type: ignore
    ported_classifier.linear_3.bias = torch.nn.Parameter(torch.from_numpy(
        keras_model.get_layer("dense_2").bias.numpy()))  # type: ignore
    torch.save(ported_classifier.state_dict(),
               BASE_CHECKPOINT_PATH.format(checkpoints_dir=checkpoints_dir, model="torch.pth"))
    return ported_classifier


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default="checkpoints")
    args = parser.parse_args()
    port_keras_model(args.checkpoints_dir)
