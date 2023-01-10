import io

import torch
from google.cloud import vision

from src.image_utils import torch_to_buffer


class GoogleNSFWModel(object):
    """
    gcloud init
    gcloud auth application-default login --no-launch-browser
    huggingface-cli login
    """
    _MAX_TRIALS = 10

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.client = vision.ImageAnnotatorClient()

    def make_request(self, image: torch.Tensor) -> vision.SafeSearchAnnotation:
        if image.ndim != 3:
            image = image.squeeze()
        if image.ndim != 3:
            raise ValueError("`make_request` can be called on individual samples only")

        with io.BytesIO() as content:
            torch_to_buffer(image, content)
            im_bytes = content.read()

        image = vision.Image(content=im_bytes)

        trial = 0
        while trial < self._MAX_TRIALS:
            try:
                response = self.client.safe_search_detection(image=image)
                safe = response.safe_search_annotation
                return safe
            except Exception as e:
                trial += 1
                if trial == self._MAX_TRIALS:
                    raise e

    def request_classification(self, image: torch.Tensor, verbose=False) -> torch.Tensor:
        response = self.make_request(image)
        if verbose:
            print(response)
        result = self.parse_results(response)
        return torch.tensor([result], device=self.device)

    def parse_results(self, response: vision.SafeSearchAnnotation) -> float:
        adult = int(response.adult)
        racy = int(response.racy)

        # medical = int(response.medical)
        # spoof = int(response.spoof)
        # violence = int(response.violence)
        return max(adult, racy) / 5.0

    def make_model_eval(self):
        pass
