import abc
from enum import Enum
import io
import json
import os
from typing import Dict, Generic, List, Optional, Set, TypeVar

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

    def __init__(self, device: torch.device, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.device = device
        if api_key is not None:
            self._API_KEY = api_key
    
    @property
    def _HEADERS(self) -> Dict[str, str]:
            return {"accept": "application/json", "authorization": f"Bearer {self._API_KEY}"}
    

    def make_request(self, image: torch.Tensor) -> ProviderResponse[ResponseLabelT]:
        if image.ndim != 3:
            image = image.squeeze()
        if image.ndim != 3:
            raise ValueError("`make_request` can be called on individual samples only")
        with io.BytesIO() as buf:
            write_torch_to_buffer(image, buf)
            data = RequestData(providers=self._PROVIDER.value)
            files = {'file': (f"image.{UPLOAD_FORMAT}", buf, f"image/{UPLOAD_FORMAT}")}
            http_response = requests.post(self._URL, data=data.dict(), files=files, headers=self._HEADERS)
        http_response.raise_for_status()
        raw_response = json.loads(http_response.text)[self._PROVIDER]
        raw_response["items"] = self._RESPONSE_LABEL_TYPE.filter_unused_labels(raw_response["items"])
        response = ProviderResponse[self._RESPONSE_LABEL_TYPE](**raw_response)  # type: ignore
        assert response.status == ResponseStatus.success
        return response

    def request_classification(self, image: torch.Tensor) -> torch.Tensor:
        response = self.make_request(image)
        result = self.parse_results(response)
        return torch.tensor([result], device=self.device)

    @abc.abstractmethod
    def parse_results(self, response: ProviderResponse[ResponseLabelT]) -> float:
        ...
        
    def make_model_eval(self):
        pass
    

def parse_all_unsafe_results(response: ProviderResponse, default_1: bool = False) -> float:
    assert response.status == ResponseStatus.success
    if default_1:
        return max(map(lambda x: x.likelihood - 1, response.items)) / 4
    else:
        return max(map(lambda x: x.likelihood, response.items)) / 5


def filter_items(response: ProviderResponse[ResponseLabelT], to_remove: Set[ResponseLabelT]) -> ProviderResponse[ResponseLabelT]:
    response.items = [item for item in response.items if item not in to_remove]
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
    _ITEMS_TO_FILTER = {GoogleResponseLabel.Medical, GoogleResponseLabel.Spoof}

    def parse_results(self, response: ProviderResponse[GoogleResponseLabel]) -> float:
        return parse_all_unsafe_results(response, default_1=True)


class ClarifaiResponseLabel(ResponseLabel):
    safe = "safe"
    drug = "drug"
    explicit = "explicit"
    suggestive = "suggestive"
    gore = "gore"


class ClarifaiNSFWModel(EdenAINSFWModel):
    _PROVIDER = Provider.clarifai
    _RESPONSE_LABEL_TYPE = ClarifaiResponseLabel

    def parse_results(self, response: ProviderResponse[ClarifaiResponseLabel]) -> float:
        labels_dict = {item.label: item.likelihood for item in response.items}
        return 1 - (labels_dict[ClarifaiResponseLabel.safe] / 5)


class MicrosoftResponseLabel(ResponseLabel):
    Adult = "Adult"
    Gore = "Gore"
    Racy = "Racy"


class MicrosoftNSFWModel(EdenAINSFWModel):
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


class AmazonNSFWModel(EdenAINSFWModel):
    _PROVIDER = Provider.amazon
    _RESPONSE_LABEL_TYPE = AmazonResponseLabel

    def parse_results(self, response: ProviderResponse[AmazonResponseLabel]) -> float:
        return parse_all_unsafe_results(response)