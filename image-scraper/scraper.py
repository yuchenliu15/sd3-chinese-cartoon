#!/usr/bin/env python3
import os
import requests
import argparse
import json
import time

BATCH_SIZE = 25 # max limit for the Real Python API is 25

def get_image_urls(num_images):
    urls = []
    for start in range(0, num_images, BATCH_SIZE):
        url = f"https://realpython.com/search/api/v1/?kind=article&kind=course&kind=topic&order=newest&continue_after={start}&limit={BATCH_SIZE}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the page: {response.status_code}")
            raise ValueError("Failed to retrieve the page")
        results = json.loads(response.text)['results']
        for result in results:
            urls.append(result['image_url'])
    print(f"get_image_urls retrieved: {urls}")
    return urls

def scrape_images(img_urls, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, img_url in enumerate(img_urls):
        try:
            img_response = requests.get(img_url)
            img_response.raise_for_status()
            img_name = os.path.join(save_dir, f"training_{i}.jpg")
            with open(img_name, 'wb') as f:
                f.write(img_response.content)
            print(f"Downloaded {img_name}")
        except requests.exceptions.RequestException as e:
            print(f"Could not download {img_url}: {e}")

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    parser = argparse.ArgumentParser(description='Scrape images from Real Python.')
    parser.add_argument('--num-images', type=int, help='The number of images to download.')
    parser.add_argument('--save-dir', type=str, help='The directory to save images.')
    
    args = parser.parse_args()
    img_urls = get_image_urls(args.num_images) 
    scrape_images(img_urls, args.save_dir)

    end_time = time.perf_counter()
    print(f"Time elapsed: {end_time - start_time:.4f} seconds")
