import json
import os
from diffusers import DiffusionPipeline
from dataset_and_utils import TokenEmbeddingsHandler
from weights import WeightsDownloadCache


class WeightsManager:
    def __init__(self, predictor):
        self.predictor = predictor
        self.weights_cache = WeightsDownloadCache()
        self.lora_scales = {}
        self.adapters = []

    def is_url(self, path):
        return path.startswith("http://") or path.startswith("https://")

    def set_lora_scales(self, scales):
        self.lora_scales.update(scales)
        self.apply_lora_scales()

    def apply_lora_scales(self):
        self.predictor.txt2img_pipe.set_adapters(self.adapters, self.lora_scales)

    def load_lora_weight(self, weight, scale, adapter_name):
        if self.is_url(weight):
            local_weights_cache = self.weights_cache.ensure(weight)
        else:
            local_weights_cache = weight

        self.predictor.txt2img_pipe.load_lora_weights(
            local_weights_cache, adapter_name=adapter_name
        )
        self.lora_scales[adapter_name] = scale
        self.adapters.append(adapter_name)
        self.apply_lora_scales()

    def load_trained_weights(self, weights_list, scales_list):
        for weights, scale in zip(weights_list, scales_list):
            adapter_name = os.path.basename(weights).split(".")[0]
            self.load_lora_weight(weights, scale, adapter_name)

        self.predictor.is_lora = True
