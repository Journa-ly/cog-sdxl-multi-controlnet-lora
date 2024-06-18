import json
import os
from diffusers import DiffusionPipeline
from dataset_and_utils import TokenEmbeddingsHandler
from weights import WeightsDownloadCache
from cog import Path


class WeightsManager:
    def __init__(self, predictor):
        self.predictor = predictor
        self.weights_cache = WeightsDownloadCache()
        self.lora_scales = {}
        self.adapters = []

    def is_url(self, path):
        return path.startswith("http://") or path.startswith("https://")

    def apply_lora_scales(self, adapter_name, scale):
        self.predictor.txt2img_pipe.set_adapters(adapter_name, scale)

    def is_adapter_loaded(self, adapter_name):
        active_adapters = self.predictor.txt2img_pipe.get_active_adapters()
        return adapter_name in active_adapters

    def load_lora_weight(self, weight, scale, adapter_name):
        if self.is_adapter_loaded(adapter_name):
            print(f"Adapter {adapter_name} is already loaded. Skipping.")
            print(
                f"DEBUG - Here are the currently loaded LoRAs: {self.predictor.txt2img_pipe.get_active_adapters()}"
            )
            return

        if self.is_url(weight):
            remote_weight = self.weights_cache.ensure(weight)
        else:
            print(f"Loading LoRA weight: {weight} with adapter name: {adapter_name}")

            # Load the LoRA weights
            self.predictor.txt2img_pipe.load_lora_weights(
                pretrained_model_name_or_path_or_dict=weight,
                adapter_name=adapter_name,
            )

            # Set the scale for the adapter
            self.apply_lora_scales(adapter_name, scale)

    def load_trained_weights(self, weight, scale, adapter_name):
        self.load_lora_weight(weight, scale, adapter_name)
        self.predictor.is_lora = True
