import os
import requests
from bs4 import BeautifulSoup
import argparse

def scrape_images(topic, num_images, save_dir):
    url = f"https://realpython.com/{topic}/"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve the page: {response.status_code}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    count = 0
    for img in img_tags:
        if count >= num_images:
            break
        img_url = img.get('src')
        if img_url:
            try:
                img_response = requests.get(img_url)
                img_response.raise_for_status()
                img_name = os.path.join(save_dir, f"{topic}_{count + 1}.jpg")
                with open(img_name, 'wb') as f:
                    f.write(img_response.content)
                count += 1
                print(f"Downloaded {img_name}")
            except requests.exceptions.RequestException as e:
                print(f"Could not download {img_url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape images from Real Python.')
    parser.add_argument('topic', type=str, help='The topic to scrape images for.')
    parser.add_argument('num_images', type=int, help='The number of images to download.')
    parser.add_argument('save_dir', type=str, help='The directory to save images.')
    
    args = parser.parse_args()
    
    scrape_images(args.topic, args.num_images, args.save_dir)