import abc
from enum import Enum
import io
import json
import os
from typing import Dict, Generic, List, Type, TypeVar

import requests
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from torchvision.transforms.functional import to_pil_image

load_dotenv()

UPLOAD_FORMAT = 'png'


def write_torch_to_buffer(image: torch.Tensor, buf: io.BytesIO) -> None:
    pil_image = to_pil_image(image)
    pil_image.save(buf, format=UPLOAD_FORMAT)
    buf.seek(0)


class Provider(str, Enum):
    google = "google"
    microsoft = "microsoft"
    amazon = "amazon"
    clarifai = "clarifai"


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


class EdenAINSFWModel(abc.ABC):
    _PROVIDER: Provider
    _URL: str = "https://api.edenai.run/v2/image/explicit_content"
    _API_KEY: str = os.environ['EDENAI_TEST_API_KEY']
    _HEADERS = {"accept": "application/json", "authorization": f"Bearer {_API_KEY}"}

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def request_classification(self, image: torch.Tensor) -> torch.Tensor:
        with io.BytesIO() as buf:
            write_torch_to_buffer(image, buf)
            data = RequestData(providers=self._PROVIDER.value)
            files = {'file': (f"image.{UPLOAD_FORMAT}", buf, f"image/{UPLOAD_FORMAT}")}
        response = requests.post(self._URL, data=data.dict(), files=files, headers=self._HEADERS)
        response.raise_for_status()
        raw_result = json.loads(response.text)[self._PROVIDER]
        result = self.parse_results(raw_result)
        return torch.Tensor([result], device=self.device)

    @abc.abstractmethod
    def parse_results(self, result: Dict) -> float:
        ...


EnumT = TypeVar("EnumT", bound=Enum)


def filter_unused_labels(result: List[Dict[str, int]], enum: Type[EnumT]) -> List[Dict[str, int]]:
    return [item for item in result if item["label"] in set(enum)]


def parse_all_unsafe_results(response: ProviderResponse) -> float:
    assert response.status == ResponseStatus.success
    return max(map(lambda x: x.likelihood, response.items)) / 5


class GoogleResponseLabel(str, Enum):
    Adult = "Adult"
    Spoof = "Spoof"
    Medical = "Medical"
    Gore = "Gore"
    Racy = "Racy"


class GoogleNSFWModel(EdenAINSFWModel):
    _PROVIDER = Provider.google

    def parse_results(self, result: Dict) -> float:
        response = ProviderResponse[GoogleResponseLabel](**result)
        return parse_all_unsafe_results(response)


class ClarifaiResponseLabel(str, Enum):
    safe = "safe"
    drug = "drug"
    explicit = "explicit"
    suggestive = "suggestive"
    gore = "gore"


class ClarifaiNSFWModel(EdenAINSFWModel):
    _PROVIDER = Provider.clarifai

    def parse_results(self, result: Dict) -> float:
        response = ProviderResponse[ClarifaiResponseLabel](**result)
        assert response.status == ResponseStatus.success
        labels_dict = {item.label: item.likelihood for item in response.items}
        return 1 - (labels_dict[ClarifaiResponseLabel.safe] / 5)


class MicrosoftResponseLabel(str, Enum):
    Adult = "Adult"
    Gore = "Gore"
    Racy = "Racy"


class MicrosoftNSFWModel(EdenAINSFWModel):
    _PROVIDER = Provider.microsoft

    def parse_results(self, result: Dict) -> float:
        response = ProviderResponse[MicrosoftResponseLabel](**result)
        assert response.status == ResponseStatus.success
        return parse_all_unsafe_results(response)


class AmazonResponseLabel(str, Enum):
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


class AmazonNSFWModel(EdenAINSFWModel):
    _PROVIDER = Provider.amazon

    def parse_results(self, result: Dict) -> float:
        result["items"] = filter_unused_labels(result["items"], AmazonResponseLabel)
        response = ProviderResponse[AmazonResponseLabel](**result)
        assert response.status == ResponseStatus.success
        return parse_all_unsafe_results(response)