import torch
from torch import nn
from transformers import CLIPModel

BASE_CHECKPOINT_PATH = "{checkpoints_dir}/clip_autokeras_nsfw_{model}"
PRETRAINED_URL_NAMES = {"b32": "base-patch32", "l14": "large-patch14"}


class PortedCLIPClassifier(nn.Module):
    DROPOUT_RATE = 0.5
    INNER_SIZE = 32

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.register_buffer("category_mapper", torch.ones(1, input_size))
        self.register_buffer("mean", torch.empty(1, input_size))
        self.register_buffer("std", torch.empty(1, input_size))
        self.linear_1 = nn.Linear(input_size, self.INNER_SIZE)
        self.linear_2 = nn.Linear(self.INNER_SIZE, self.INNER_SIZE)
        self.linear_3 = nn.Linear(self.INNER_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = x * self.category_mapper
        x = (x - self.mean) / self.std
        x = nn.functional.relu(self.linear_1(x))
        x = nn.functional.relu(self.linear_2(x))
        # for some reason this breaks the model equivalency tests even in eval mode
        # x = nn.functional.dropout(x, self.DROPOUT_RATE)
        x = self.linear_3(x)
        return torch.sigmoid(x)


class CLIPNSFWDetector(nn.Module):
    INPUT_SIZE = 512

    def __init__(self, model: str = "b32", checkpoints_dir: str = "checkpoints"):
        super().__init__()
        checkpoint_path = BASE_CHECKPOINT_PATH.format(checkpoints_dir=checkpoints_dir, model="torch.pth")
        self.clip_embedder = CLIPModel.from_pretrained(f"openai/clip-vit-{PRETRAINED_URL_NAMES[model]}")
        self.nsfw_model = PortedCLIPClassifier(self.INPUT_SIZE)
        self.nsfw_model.load_state_dict(torch.load(checkpoint_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        embeddings = self.clip_embedder.get_image_features(x)  # type: ignore
        normalized_embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        y = self.nsfw_model(normalized_embeddings)
        return y


def port_keras_model(checkpoints_dir: str = "checkpoints") -> PortedCLIPClassifier:
    import autokeras as ak
    import tensorflow as tf
    ported_classifier = PortedCLIPClassifier(512)
    keras_model: tf.keras.Model = tf.keras.models.load_model(BASE_CHECKPOINT_PATH.format(
        checkpoints_dir=checkpoints_dir, model="b32"),
                                                             custom_objects=ak.CUSTOM_OBJECTS)
    encoding_layers = keras_model.get_layer("multi_category_encoding").encoding_layers
    category_mapper = torch.tensor([[1. if layer is None else 0. for layer in encoding_layers]])
    ported_classifier.category_mapper = category_mapper
    ported_classifier.mean = torch.from_numpy(keras_model.get_layer("normalization").mean.numpy())
    ported_classifier.std = torch.sqrt(torch.from_numpy(keras_model.get_layer("normalization").variance.numpy()))
    ported_classifier.linear_1.weight = torch.nn.Parameter(
        torch.from_numpy(keras_model.get_layer("dense").kernel.numpy()).t())
    ported_classifier.linear_1.bias = torch.nn.Parameter(torch.from_numpy(keras_model.get_layer("dense").bias.numpy()))
    ported_classifier.linear_2.weight = torch.nn.Parameter(
        torch.from_numpy(keras_model.get_layer("dense_1").kernel.numpy()).t())
    ported_classifier.linear_2.bias = torch.nn.Parameter(torch.from_numpy(
        keras_model.get_layer("dense_1").bias.numpy()))
    ported_classifier.linear_3.weight = torch.nn.Parameter(
        torch.from_numpy(keras_model.get_layer("dense_2").kernel.numpy()).t())
    ported_classifier.linear_3.bias = torch.nn.Parameter(torch.from_numpy(
        keras_model.get_layer("dense_2").bias.numpy()))
    torch.save(ported_classifier.state_dict(),
               BASE_CHECKPOINT_PATH.format(checkpoints_dir=checkpoints_dir, model="torch.pth"))
    return ported_classifier
