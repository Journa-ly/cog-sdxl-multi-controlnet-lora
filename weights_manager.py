from diffusers.models.attention_processor import AttnProcessor2_0
from diffusers.utils import LoraLoaderMixin


class WeightsManager:
    def __init__(self, predictor):
        self.predictor = predictor
        self.weights_cache = WeightsDownloadCache()

    def is_url(self, path):
        return path.startswith("http://") or path.startswith("https://")

    def load_trained_weights(self, weights, pipe, scale=1.0):
        from no_init import no_init_or_tensor

        weights = str(weights)
        if self.predictor.tuned_weights == weights:
            print("Skipping loading... weights already loaded")
            return

        self.predictor.tuned_weights = weights

        if self.is_url(weights):
            local_weights_cache = self.weights_cache.ensure(weights)
        else:
            local_weights_cache = weights

        # Check if the local_weights_cache is a direct file path to LoRA weights
        if local_weights_cache.endswith(".safetensors"):
            print("Loading Unet LoRA from specific .safetensors file")

            unet = pipe.unet
            print(f"loading LoRA: {local_weights_cache}")

            # Load LoRA weights using the recommended method
            LoraLoaderMixin.load_lora_weights(unet, local_weights_cache)

            # Apply scale to LoRA layers
            for name, module in unet.named_modules():
                if isinstance(module, AttnProcessor2_0):
                    module.set_lora_scale(scale)

            self.predictor.is_lora = True

            # Set an empty token_map if loading directly from .safetensors file
            self.predictor.token_map = {}

        else:
            # Load UNET
            print("Loading fine-tuned model")
            self.predictor.is_lora = False

            maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
            if not os.path.exists(maybe_unet_path):
                print("Does not have Unet. Assume we are using LoRA")
                self.predictor.is_lora = True

            if not self.predictor.is_lora:
                print("Loading Unet")

                new_unet_params = load_file(
                    os.path.join(local_weights_cache, "unet.safetensors")
                )
                pipe.unet.load_state_dict(new_unet_params, strict=False)

            else:
                print("Loading Unet LoRA")

                unet = pipe.unet

                # Load LoRA weights using the recommended method
                LoraLoaderMixin.load_lora_weights(
                    unet, os.path.join(local_weights_cache, "lora.safetensors")
                )

                # Apply scale to LoRA layers
                for name, module in unet.named_modules():
                    if isinstance(module, AttnProcessor2_0):
                        module.set_lora_scale(scale)

            # Load text
            handler = TokenEmbeddingsHandler(
                [pipe.text_encoder, pipe.text_encoder_2],
                [pipe.tokenizer, pipe.tokenizer_2],
            )
            handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

            # Load params
            with open(
                os.path.join(local_weights_cache, "special_params.json"), "r"
            ) as f:
                params = json.load(f)
            self.predictor.token_map = params

        self.predictor.tuned_model = True
