import autokeras as ak
import numpy as np
import tensorflow as tf
import torch

from src.arch.clip_laion_nsfw import PortedCLIPClassifier


def test_ported_model():
    total_tests = 512
    checkpoint_path = "checkpoints/clip_autokeras_nsfw_b32"
    original_model: tf.keras.Model = tf.keras.models.load_model(checkpoint_path,
                                                                custom_objects=ak.CUSTOM_OBJECTS)  # type: ignore
    ported_model = PortedCLIPClassifier(512).eval()
    ported_model.load_state_dict(torch.load("checkpoints/clip_autokeras_nsfw_torch.pth"))

    np.testing.assert_allclose(
        original_model.get_layer("normalization").mean.numpy(),
        ported_model.mean.detach().numpy(),  # type: ignore
        rtol=1e-5,
        atol=1e-5)
    np.testing.assert_allclose(
        original_model.get_layer("normalization").variance.numpy(),
        ported_model.std.detach().numpy()**2,  # type: ignore
        rtol=1e-5,
        atol=1e-5)
    np.testing.assert_allclose(original_model.get_layer("dense").kernel.numpy(),
                               ported_model.linear_1.weight.detach().numpy().T,
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(original_model.get_layer("dense").bias.numpy(),
                               ported_model.linear_1.bias.detach().numpy(),
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(original_model.get_layer("dense_1").kernel.numpy(),
                               ported_model.linear_2.weight.detach().numpy().T,
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(original_model.get_layer("dense_1").bias.numpy(),
                               ported_model.linear_2.bias.detach().numpy(),
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(original_model.get_layer("dense_2").kernel.numpy(),
                               ported_model.linear_3.weight.detach().numpy().T,
                               rtol=1e-5,
                               atol=1e-5)
    np.testing.assert_allclose(original_model.get_layer("dense_2").bias.numpy(),
                               ported_model.linear_3.bias.detach().numpy(),
                               rtol=1e-5,
                               atol=1e-5)

    mean = ported_model.mean.numpy()  # type: ignore
    std = ported_model.std.numpy()  # type: ignore
    np.random.seed(0)
    random_input = np.random.normal(loc=mean, scale=std, size=(total_tests, 512))

    original_model_output = original_model.predict(random_input)

    pt_input = torch.from_numpy(random_input).float()
    pt_output = ported_model(pt_input).detach().numpy()  # type: ignore

    np.testing.assert_allclose(original_model_output, pt_output, rtol=1e-5, atol=1e-5)