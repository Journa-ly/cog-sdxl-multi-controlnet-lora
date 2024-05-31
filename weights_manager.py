import json
import os
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from safetensors.torch import load_file
from dataset_and_utils import TokenEmbeddingsHandler
from weights import WeightsDownloadCache


class WeightsManager:
    def __init__(self, predictor):
        self.predictor = predictor
        self.weights_cache = WeightsDownloadCache()

    def is_url(self, path):
        return path.startswith("http://") or path.startswith("https://")

    def load_trained_weights(self, weights, pipe):
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
            tensors = load_file(local_weights_cache)

            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                with no_init_or_tensor():
                    module = LoRAAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=name_rank_map.get(
                            name, 4
                        ),  # Default to rank 4 if not found
                    )
                unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

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

                tensors = load_file(
                    os.path.join(local_weights_cache, "lora.safetensors")
                )

                unet_lora_attn_procs = {}
                name_rank_map = {}
                for tk, tv in tensors.items():
                    if tk.endswith("up.weight"):
                        proc_name = ".".join(tk.split(".")[:-3])
                        r = tv.shape[1]
                        name_rank_map[proc_name] = r

                for name, attn_processor in unet.attn_processors.items():
                    cross_attention_dim = (
                        None
                        if name.endswith("attn1.processor")
                        else unet.config.cross_attention_dim
                    )
                    if name.startswith("mid_block"):
                        hidden_size = unet.config.block_out_channels[-1]
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(reversed(unet.config.block_out_channels))[
                            block_id
                        ]
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks.")])
                        hidden_size = unet.config.block_out_channels[block_id]
                    with no_init_or_tensor():
                        module = LoRAAttnProcessor2_0(
                            hidden_size=hidden_size,
                            cross_attention_dim=cross_attention_dim,
                            rank=name_rank_map.get(
                                name, 4
                            ),  # Default to rank 4 if not found
                        )
                    unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)

                unet.set_attn_processor(unet_lora_attn_procs)
                unet.load_state_dict(tensors, strict=False)

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
