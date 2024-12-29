import cv2
import os
import json

CONFIG = {
    "dataset_folder": r"C:\\Object Model Classification\\datasets",
    "output_folder": "C:\\Object Model Classification\\processed_videos",
    "min_contour_area": 500,
    "model_state_file": "mog2_state.npy"
}

config_file = "main.json"
with open(config_file, "w") as f:
    json.dump(CONFIG, f, indent=4)

with open(config_file, "r") as f:
    config = json.load(f)

dataset_folder = config["dataset_folder"]
output_folder = config["output_folder"]
min_contour_area = config["min_contour_area"]
model_state_file = config["model_state_file"]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

if os.path.exists(model_state_file):
    bg_subtractor = cv2.BackgroundSubtractorMOG2.create()
    bg_subtractor.load(model_state_file)

car_count = 0
frame_number = 0

print(f"Starting video processing...")

for video_file in os.listdir(dataset_folder):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(dataset_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video file: {video_file}")
        continue

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(output_folder, f"output_{video_file}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {video_file}, width: {frame_width}, height: {frame_height}, fps: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video: {video_file}")
            break

        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"Frame {frame_number}: Found {len(contours)} contours")

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            car_count += 1

        out.write(frame)

        frame_number += 1

    cap.release()
    out.release()

print(f"Saving background subtractor model to {model_state_file}...")
bg_subtractor.save(model_state_file)

print(f"All videos processed. Total cars detected: {car_count}")
