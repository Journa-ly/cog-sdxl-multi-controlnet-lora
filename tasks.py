import os
from celery import Celery, shared_task
from predict import Predictor

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "5672")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

app = Celery(
    'tasks',
    broker=f"amqp://{RABBITMQ_USER}:{RABBITMQ_PASS}@{RABBITMQ_HOST}:{RABBITMQ_PORT}"
)

@shared_task(bind=True, queue="gqu_enabled_queue")
def generate_image(self, payload, **kwargs):
    command = [
        "cog", "predict",
        "-i", f"prompt={prompt}",
        "-i", f"negative_prompt={negative_prompt}",
        "-i", f"image={str(image) if image else ''}",
        "-i", f"mask={str(mask) if mask else ''}",
        "-i", f"width={width}",
        "-i", f"height={height}",
        "-i", f"sizing_strategy={sizing_strategy}",
        "-i", f"num_outputs={num_outputs}",
        "-i", f"sdxl_checkpoint={sdxl_checkpoint}",
        "-i", f"scheduler={scheduler}",
        "-i", f"num_inference_steps={num_inference_steps}",
        "-i", f"guidance_scale={guidance_scale}",
        "-i", f"prompt_strength={prompt_strength}",
        "-i", f"seed={seed if seed else ''}",
        "-i", f"refine={refine}",
        "-i", f"refine_steps={refine_steps if refine_steps else ''}",
        "-i", f"apply_watermark={apply_watermark}",
        "-i", f"base_lora_ID={base_lora_ID}",
        "-i", f"base_lora_path={base_lora_path}",
        "-i", f"base_lora_scale={base_lora_scale}",
        "-i", f"style_lora_ID={style_lora_ID}",
        "-i", f"style_lora_path={style_lora_path}",
        "-i", f"style_lora_scale={style_lora_scale}",
        "-i", f"disable_safety_checker={disable_safety_checker}",
        "-i", f"controlnet_1={controlnet_1}",
        "-i", f"controlnet_1_image={str(controlnet_1_image) if controlnet_1_image else ''}",
        "-i", f"controlnet_1_conditioning_scale={controlnet_1_conditioning_scale}",
        "-i", f"controlnet_1_start={controlnet_1_start}",
        "-i", f"controlnet_1_end={controlnet_1_end}",
        "-i", f"controlnet_2={controlnet_2}",
        "-i", f"controlnet_2_image={str(controlnet_2_image) if controlnet_2_image else ''}",
        "-i", f"controlnet_2_conditioning_scale={controlnet_2_conditioning_scale}",
        "-i", f"controlnet_2_start={controlnet_2_start}",
        "-i", f"controlnet_2_end={controlnet_2_end}",
        "-i", f"controlnet_3={controlnet_3}",
        "-i", f"controlnet_3_image={str(controlnet_3_image) if controlnet_3_image else ''}",
        "-i", f"controlnet_3_conditioning_scale={controlnet_3_conditioning_scale}",
        "-i", f"controlnet_3_start={controlnet_3_start}",
        "-i", f"controlnet_3_end={controlnet_3_end}",
    ]

    # Filter out empty arguments
    command = [arg for arg in command if arg]

    output = subprocess.run(command, capture_output=True, text=True)
    return output