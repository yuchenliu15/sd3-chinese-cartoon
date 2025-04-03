import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

OUTPUT_DIR = '../training_extracted_images_blip'
IMAGE_DIR = '../training_extracted_images'

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("mps")

os.makedirs(OUTPUT_DIR, exist_ok=True)


image_files = [f for f in os.listdir(IMAGE_DIR)]

for file in image_files:
    image = Image.open(os.path.join(IMAGE_DIR, file)).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    with open(os.path.join(OUTPUT_DIR, file.replace('.jpg', '.txt')), 'w') as f:
        f.write(caption)
    print(f"finished: {file}")

