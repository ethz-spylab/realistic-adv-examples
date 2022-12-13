import io

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

UPLOAD_FORMAT = 'png'


def encode_decode(image: torch.Tensor, format=UPLOAD_FORMAT) -> torch.Tensor:
    with io.BytesIO() as buf:
        torch_to_buffer(image.squeeze(), buf, format)
        png_image = buffer_to_torch(buf, image.device, format).unsqueeze(0)
    out = png_image
    return out


def to_from_pil(image):
    pil_image = Image.fromarray(np.uint8(torch.round(image[0] * 255).cpu().numpy().transpose(1, 2, 0)))
    np_image = (np.asarray(pil_image).astype(np.float32)).transpose(2, 0, 1)
    converted_image = torch.from_numpy(np_image).to(image.device) / 255
    return converted_image.unsqueeze(0)


def buffer_to_torch(buf: io.BytesIO, device: torch.device, format=UPLOAD_FORMAT) -> torch.Tensor:
    image = Image.open(buf, formats=[format])
    image.load()
    np_image = (np.asarray(image).astype(np.float32)).transpose(2, 0, 1)
    torch_image = torch.from_numpy(np_image).to(device) / 255
    assert torch.allclose(torch_image * 255, pil_to_tensor(image).to(device).to(torch.float), atol=1 / 256)
    return torch_image


def torch_to_buffer(image: torch.Tensor, buf: io.BytesIO, format=UPLOAD_FORMAT) -> None:
    pil_image = Image.fromarray(np.uint8(torch.round(image * 255).cpu().numpy().transpose(1, 2, 0)))
    if format == 'png':
        pil_image.save(buf, format=format, compress_level=0, optimize=False)
    else:
        pil_image.save(buf, format=format)
    # pil_image.save("test.png")
    buf.seek(0)
