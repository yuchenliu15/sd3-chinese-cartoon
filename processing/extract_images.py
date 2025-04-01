import os
import cv2

def main():
    video_path = '../videos/danaotiangong.mp4'
    output_path = '../training_extracted_images'
    num_images = 200

    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return

    os.makedirs(output_path, exist_ok=True)

    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    interval = total_frames // num_images

    frame_count = 0
    wrote_count = 0
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret or frame_count >= total_frames:
            break
        if frame_count % interval == 0:
            image_path = os.path.join(output_path, f"training_{wrote_count}.jpg")
            cv2.imwrite(image_path, frame)
            print("Wrote image:", image_path)
            wrote_count += 1
        frame_count += 1
    capture.release()

if __name__ == "__main__":
    main()