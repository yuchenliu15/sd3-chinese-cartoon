from diffusers import AutoPipelineForText2Image
import torch
import os

MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
#pipeline = StableDiffusion3Pipeline.from_pretrained(
#    args.pretrained_model_name_or_path,
#    revision=args.revision,
#    variant=args.variant,
#    torch_dtype=weight_dtype,
#)

# load attention processors
#pipeline.load_lora_weights(args.output_dir)

pipeline = AutoPipelineForText2Image.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to('mps')
pipeline.load_lora_weights(os.path.dirname('../finetuned_output/'), weight_name='pytorch_lora_weights.safetensors')
image = pipeline('tiger on the mountain surrounded by clouds in cartoon style').images[0]
image.save('output1.png')
