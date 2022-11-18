import autokeras as ak
import tensorflow as tf
from transformers import TFCLIPModel

BASE_CHECKPOINT_PATH = "{checkpoints_dir}/clip_autokeras_nsfw_{model}"
PRETRAINED_URL_NAMES = {"b32": "base-patch32", "l14": "large-patch14"}


class CLIPNSFWDetector(tf.keras.Model):

    def __init__(self, model: str = "b32", checkpoints_dir: str = "checkpoints"):
        super().__init__()
        checkpoint_path = BASE_CHECKPOINT_PATH.format(checkpoints_dir=checkpoints_dir, model=model)
        self.clip_embedder = TFCLIPModel.from_pretrained(f"openai/clip-vit-{PRETRAINED_URL_NAMES[model]}")
        self.nsfw_model: tf.keras.Model = tf.keras.models.load_model(checkpoint_path,
                                                                     custom_objects=ak.CUSTOM_OBJECTS)  # type: ignore

    def call(self, inputs, training=None, mask=None):
        embeddings = self.clip_embedder.get_image_features(inputs)
        normalized_embeddings, _ = tf.linalg.normalize(embeddings, axis=1)
        y = self.nsfw_model(normalized_embeddings)
        return y
