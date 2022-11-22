import abc
from dataclasses import dataclass
from enum import Enum
import io
import json
import os
from typing import Dict, Generic, List, TypeVar

import requests
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel
from torchvision.transforms.functional import to_pil_image

load_dotenv()


UPLOAD_FORMAT = 'JPEG'


def torch_to_binary(image: torch.Tensor) -> io.BytesIO:
    pil_image = to_pil_image(image)
    buf = io.BytesIO()
    pil_image.save(buf, format=UPLOAD_FORMAT)
    buf.seek(0)
    return buf


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
        binary_image = torch_to_binary(image)
        data = RequestData(providers=self._PROVIDER.value)
        files = {'file': (f"image.{UPLOAD_FORMAT}", binary_image, f"image/{UPLOAD_FORMAT}")}
        response = requests.post(self._URL, data=data.dict(), files=files, headers=self._HEADERS)
        response.raise_for_status()
        raw_result = json.loads(response.text)[self._PROVIDER]
        result = self.parse_results(raw_result)
        return torch.Tensor([result], device=self.device)

    @abc.abstractmethod
    def parse_results(self, result: Dict) -> float:
        ...


class GoogleResponseLabel(str, Enum):
    Adult = "Adult"
    Spoof = "Spoof"
    Medical = "Medical"
    Gore = "Gore"
    Racy = "Racy"


class GoogleNSFWModel(EdenAINSFWModel):
    _PROVIDER = Provider.google

    def parse_results(self, result: Dict) -> float:
        print(result)
        response = ProviderResponse[GoogleResponseLabel](**result)
        assert response.status == ResponseStatus.success
        return max(map(lambda x: x.likelihood, response.items)) / 5
