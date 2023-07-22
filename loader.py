import torch
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from enum import Enum
from typing import Optional
import json

class Loader:
    class Module(Enum):
        IMAGE_TO_IMAGE = "img2img"

    def __init__(self):
        self._pipes = {
            Loader.Module.IMAGE_TO_IMAGE: {}
        }
        self._model_ids = {}
        self._embeddings = {}
        self._load_model_ids()
        self._load_pipe()

    def _load_model_ids(self):
        with open("model_ids.json", 'r') as json_file:
            self._model_ids = json.load(json_file)

        with open("embeddings.json", 'r') as json_file:
            self._embeddings = json.load(json_file)

    def _load_pipe(self, module: Optional["Loader.Module"] = None, model: Optional[str] = None):
        for index, (name, id) in enumerate(self._model_ids.items()):
            sch = EulerAncestralDiscreteScheduler.from_pretrained(id, subfolder = "scheduler")
            self._pipes[Loader.Module.IMAGE_TO_IMAGE][name] = StableDiffusionImg2ImgPipeline.from_pretrained(id, scheduler = sch, torch_dtype = torch.float16).to("cuda")
            for i in self._embeddings.keys():
                self._pipes[Loader.Module.IMAGE_TO_IMAGE][name].load_textual_inversion(self._embeddings[i], token = i)

            
    def get_pipe(self, module: "Loader.Module", model: str):
        return self._pipes[module][model]