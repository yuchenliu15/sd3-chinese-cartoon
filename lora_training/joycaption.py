#!/usr/bin/env python3

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import pathlib

PROMPT = "Write a long descriptive caption for this image in a formal tone."
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
BATCH_SIZE = 50 

def process_images(output_dir: str, images_path: list[str]):
    print("process_images args", output_dir, images_path)
    for i in range(0, len(images_path), BATCH_SIZE):
        process_batch(output_dir, i, i + BATCH_SIZE, images_path)

def process_batch(output_dir: str, start, end, images_path: list[str]):
    print("process_batch args", output_dir, start, end, images_path)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    llava_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="bfloat16", device_map=0)
    llava_model.eval()

    with torch.no_grad():
        # Load image
        images = [Image.open(images_path[i]) for i in range(start, end)]

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful cartoon image captioner.",
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
        )

        # Trim off the prompt
        generate_ids = generate_ids[:,inputs['input_ids'].shape[1]:]

        # Decode the caption
        captions = processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        write_captions(output_dir, images_path, captions, start, end)

def get_files_in_dir(directory: str):
    return [str(p) for p in pathlib.Path(directory).rglob("*") if p.is_file()]

def write_captions(output_dir: str, output_file_names: list[str], captions: list[str], start, end):
    dir_path = pathlib.Path(output_dir)
    dir_path.mkdir(exist_ok=True)
    if len(captions) != end-start:
        raise ValueError(f"Number of captions ({len(captions)}) !=  number of images ({end-start})")
    for i in range(len(captions)):
        file_name, caption = output_file_names[start+i], captions[i]
        with open(dir_path / f"{pathlib.Path(file_name).stem}.txt", "w") as f:
            f.write(caption.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JoyCaption: Image Captioning with Llama 3.1")
    parser.add_argument("--image", type=str, help="Path to the image to caption")
    parser.add_argument("--dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--output-dir", required=True, type=str, default=None, help="Directory to save the captioned images")
    args = parser.parse_args()
    all_images = [f for f in get_files_in_dir(args.dir)]
    if args.image:
        all_images.append(args.image)
    if not all_images:
        raise ValueError("No images found")
    process_images(args.output_dir, all_images)
