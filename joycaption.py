#!/usr/bin/env python3

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import pathlib

PROMPT = "Write a long descriptive caption for this image in a formal tone."
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

def process_images(output: str, images_path: list[str]):
    # Load JoyCaption
    # bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1)
    # device_map=0 loads the model into the first GPU
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    llava_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="bfloat16", device_map=0)
    llava_model.eval()

    with torch.no_grad():
        # Load image
        images = [Image.open(image_path) for image_path in images_path]

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": PROMPT,
            },
        ]

        # Format the conversation
        # WARNING: HF's handling of chat's on Llava models is very fragile.  This specific combination of processor.apply_chat_template(), and processor() works
        # but if using other combinations always inspect the final input_ids to ensure they are correct.  Often times you will end up with multiple <bos> tokens
        # if not careful, which can make the model perform poorly.
        convo_string = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = processor(text=[convo_string]*len(images), images=images, return_tensors="pt").to('mps')
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Generate the captions
        generate_ids = llava_model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            suppress_tokens=None,
            use_cache=True,
            temperature=0.6,
            top_k=None,
            top_p=0.9,
        )[0]

        # Trim off the prompt
        generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

        # Decode the caption
        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.strip()
        print(inputs)
        print()
        print(generate_ids)
        print()
        print(caption)
        #with open(output, "w") as f:
        #    f.write(caption)

def get_files_in_dir(directory: str):
    return [str(p) for p in pathlib.Path(directory).rglob("*") if p.is_file()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JoyCaption: Image Captioning with Llama 3.1")
    parser.add_argument("--image", type=str, help="Path to the image to caption")
    parser.add_argument("--dir", type=str, default=None, help="Directory of images")
    parser.add_argument("-output", type=str, default=None, help="Path to save the captioned image")
    args = parser.parse_args()
    all_images = [f for f in get_files_in_dir(args.dir)]
    if args.image:
        all_images.append(args.image)
    process_images(args.output, all_images)
