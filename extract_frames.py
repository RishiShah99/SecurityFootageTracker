# This code will extract frames from a video file and save them as images in a folder
import cv2
import os

def extract_frames(video_file, output_folder):
    cap = cv2.VideoCapture(video_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_file = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_file, frame)

    cap.release()

# Example usage
video_file = "Video_1.mp4"
output_folder = "frames"
extract_frames(video_file, output_folder)
