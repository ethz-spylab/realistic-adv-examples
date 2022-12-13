import abc
from enum import Enum
import io
import json
import os
import warnings
from typing import Dict, Generic, List, Optional, Set, TypeVar

import requests
import tensorflow as tf
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from torchvision import transforms

from src.arch.clip_laion_nsfw import CLIPNSFWDetector
from src.image_utils import encode_decode, torch_to_buffer, UPLOAD_FORMAT

load_dotenv()


class Provider(str, Enum):
    google = "google"
    microsoft = "microsoft"
    amazon = "amazon"
    clarifai = "clarifai"
    api4ai = "api4ai"
    laion = "laion"


T = TypeVar("T")


class ResponseStatus(str, Enum):
    success = "success"


class ResponseItem(GenericModel, Generic[T]):
    label: T
    likelihood: int = Field(...)


class ProviderResponse(GenericModel, Generic[T]):
    status: ResponseStatus = Field(...)
    items: List[ResponseItem[T]]


class RequestData(BaseModel):
    providers: str
    attributes_as_list: bool = False
    show_original_response: bool = False
    response_as_dict: bool = True


class ResponseLabel(str, Enum):
    pass

    @classmethod
    def filter_unused_labels(cls, items: List[Dict[str, int]]) -> List[Dict[str, int]]:
        return [item for item in items if item["label"] in set(cls)]


ResponseLabelT = TypeVar("ResponseLabelT", bound=ResponseLabel)


class EdenAINSFWModel(abc.ABC, Generic[ResponseLabelT]):
    _PROVIDER: Provider
    _URL: str = "https://api.edenai.run/v2/image/explicit_content"
    _API_KEY: str = os.environ['EDENAI_TEST_API_KEY']
    _RESPONSE_LABEL_TYPE: type[ResponseLabelT]
    _MAX_TRIALS = 10

    def __init__(self, device: torch.device, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.device = device
        if api_key is not None:
            self._API_KEY = api_key

    @property
    def _HEADERS(self) -> Dict[str, str]:
        return {"accept": "application/json", "authorization": f"Bearer {self._API_KEY}"}

    def make_request(self, image: torch.Tensor, trials=0) -> ProviderResponse[ResponseLabelT]:
        if image.ndim != 3:
            image = image.squeeze()
        if image.ndim != 3:
            raise ValueError("`make_request` can be called on individual samples only")
        with io.BytesIO() as buf:
            torch_to_buffer(image, buf)
            data = RequestData(providers=self._PROVIDER.value)
            files = {'file': (f"image.{UPLOAD_FORMAT}", buf, f"image/{UPLOAD_FORMAT}")}
            http_response = requests.post(self._URL, data=data.dict(), files=files, headers=self._HEADERS)
        http_response.raise_for_status()
        raw_response = json.loads(http_response.text)[self._PROVIDER]
        if raw_response["status"] != "success":
            if trials < self._MAX_TRIALS:
                warnings.warn(
                    RuntimeWarning(f"The request failed {trials + 1} times, retrying. Response: {raw_response}"))
                return self.make_request(image, trials + 1)
            else:
                raise RuntimeError(f"The request failed too many times: {raw_response}")
        raw_response["items"] = self._RESPONSE_LABEL_TYPE.filter_unused_labels(raw_response["items"])
        response = ProviderResponse[self._RESPONSE_LABEL_TYPE](**raw_response)  # type: ignore
        assert response.status == ResponseStatus.success
        return response

    def request_classification(self, image: torch.Tensor, verbose=False) -> torch.Tensor:
        response = self.make_request(image)
        if verbose:
            print(response)
        result = self.parse_results(response)
        return torch.tensor([result], device=self.device)

    @abc.abstractmethod
    def parse_results(self, response: ProviderResponse[ResponseLabelT]) -> float:
        ...

    def get_response_items_as_dict(self, response: ProviderResponse[ResponseLabelT]) -> Dict[ResponseLabelT, int]:
        return {item.label: item.likelihood for item in response.items}

    def make_model_eval(self):
        pass


def parse_all_unsafe_results(response: ProviderResponse, default_1: bool = False) -> float:
    assert response.status == ResponseStatus.success
    if default_1:
        return max(map(lambda x: x.likelihood - 1, response.items)) / 4
    else:
        return max(map(lambda x: x.likelihood, response.items)) / 5


def filter_items(response: ProviderResponse[ResponseLabelT],
                 to_remove: Set[ResponseLabelT]) -> ProviderResponse[ResponseLabelT]:
    response.items = [item for item in response.items if item.label not in to_remove]
    return response


class GoogleResponseLabel(ResponseLabel):
    Adult = "Adult"
    Spoof = "Spoof"
    Medical = "Medical"
    Gore = "Gore"
    Racy = "Racy"


class GoogleNSFWModel(EdenAINSFWModel[GoogleResponseLabel]):
    _PROVIDER = Provider.google
    _RESPONSE_LABEL_TYPE = GoogleResponseLabel
    _ITEMS_TO_FILTER = {
        GoogleResponseLabel.Medical, GoogleResponseLabel.Spoof, GoogleResponseLabel.Gore, GoogleResponseLabel.Adult
    }

    def parse_results(self, response: ProviderResponse[GoogleResponseLabel]) -> float:
        filtered_response = filter_items(response, self._ITEMS_TO_FILTER)
        return parse_all_unsafe_results(filtered_response, default_1=True)


class ClarifaiResponseLabel(ResponseLabel):
    safe = "safe"
    drug = "drug"
    explicit = "explicit"
    suggestive = "suggestive"
    gore = "gore"


class ClarifaiNSFWModel(EdenAINSFWModel[ClarifaiResponseLabel]):
    _PROVIDER = Provider.clarifai
    _RESPONSE_LABEL_TYPE = ClarifaiResponseLabel

    def parse_results(self, response: ProviderResponse[ClarifaiResponseLabel]) -> float:
        labels_dict = {item.label: item.likelihood for item in response.items}
        return 1 - (labels_dict[ClarifaiResponseLabel.safe] / 5)


class MicrosoftResponseLabel(ResponseLabel):
    Adult = "Adult"
    Gore = "Gore"
    Racy = "Racy"


class MicrosoftNSFWModel(EdenAINSFWModel[MicrosoftResponseLabel]):
    _PROVIDER = Provider.microsoft
    _RESPONSE_LABEL_TYPE = MicrosoftResponseLabel

    def parse_results(self, response: ProviderResponse[MicrosoftResponseLabel]) -> float:
        return parse_all_unsafe_results(response)


class AmazonResponseLabel(ResponseLabel):
    ExplicitNudity = "Explicit Nudity"
    Suggestive = "Suggestive"
    Violence = "Violence"
    VisuallyDisturbing = "Visually Disturbing"
    RudeGestures = "Rude Gestures"
    Drugs = "Drugs"
    Tobacco = "Tobacco"
    Alcohol = "Alcohol"
    Gambling = "Gambling"
    HateSymbols = "Hate Symbols"


class AmazonNSFWModel(EdenAINSFWModel[AmazonResponseLabel]):
    _PROVIDER = Provider.amazon
    _RESPONSE_LABEL_TYPE = AmazonResponseLabel
    _ITEMS_TO_FILTER = set(AmazonResponseLabel) - {AmazonResponseLabel.Suggestive}
    _NEUTRAL_RESPONSE_ITEM = ResponseItem[AmazonResponseLabel](label=AmazonResponseLabel.Suggestive, likelihood=0)

    def parse_results(self, response: ProviderResponse[AmazonResponseLabel]) -> float:
        filtered_response = filter_items(response, self._ITEMS_TO_FILTER)
        if len(filtered_response.items) == 0:
            filtered_response.items = [self._NEUTRAL_RESPONSE_ITEM]
        return parse_all_unsafe_results(response)


class API4AIResponseLabel(ResponseLabel):
    nsfw = "nsfw"
    sfw = "sfw"


class API4AINSFWModel(EdenAINSFWModel[API4AIResponseLabel]):
    _PROVIDER = Provider.api4ai
    _RESPONSE_LABEL_TYPE = API4AIResponseLabel

    def parse_results(self, response: ProviderResponse[API4AIResponseLabel]) -> float:
        nsfw_likelihood = self.get_response_items_as_dict(response)[API4AIResponseLabel.nsfw]
        return (nsfw_likelihood - 1) / 4


class LAIONResponseLabel(ResponseLabel):
    nsfw = "nsfw"


class LAIONNSFWModel(EdenAINSFWModel[LAIONResponseLabel]):
    _PROVIDER = Provider.laion
    _RESPONSE_LABEL_TYPE = LAIONResponseLabel

    _MEAN_LIST = [0.48145466, 0.4578275, 0.40821073]
    _STD_LIST = [0.26862954, 0.26130258, 0.27577711]

    def __init__(self, device: torch.device, api_key: Optional[str] = None, strong_preprocessing=True) -> None:
        super().__init__(device, api_key)
        self.model = CLIPNSFWDetector()
        self.mean = torch.Tensor(self._MEAN_LIST).view(1, 3, 1, 1).to(device)
        self.std = torch.Tensor(self._STD_LIST).view(1, 3, 1, 1).to(device)
        self.apply_preprocessing = strong_preprocessing
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])

    def make_request(self, image: torch.Tensor, trials=0) -> ProviderResponse[LAIONResponseLabel]:
        if self.apply_preprocessing:
            image = encode_decode(image, format='png')
            image = self.transform(image)
        else:
            image = encode_decode(image, format='png')
        image = (image - self.mean) / self.std
        image_tf = tf.constant(image.cpu())
        likelihood = round(self.model(image_tf).numpy().item() * 5)
        response_item = ResponseItem(label=LAIONResponseLabel.nsfw, likelihood=likelihood)
        return ProviderResponse[LAIONResponseLabel](status=ResponseStatus.success, items=[response_item])

    def parse_results(self, response: ProviderResponse[LAIONResponseLabel]) -> float:
        nsfw_likelihood = self.get_response_items_as_dict(response)[LAIONResponseLabel.nsfw]
        return nsfw_likelihood / 5
