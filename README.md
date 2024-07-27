# Security Footage Tracker

## Introduction
The Security Footage Tracker is a Python-based project that performs pose estimation on video footage. The system extracts frames from a video, applies a pre-trained Keypoint R-CNN model to identify human keypoints, and generates an output video with the keypoints and connections overlaid. This tool is useful for analyzing security footage to detect human movements and poses.

## Features
- Extract frames from video files
- Perform pose estimation on each frame using Keypoint R-CNN
- Overlay keypoints and connections on frames
- Generate a processed video with pose estimation visualizations

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- matplotlib
- tqdm
- PIL (Pillow)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SecurityFootageTracker.git
   cd SecurityFootageTracker
    ```
2. Install the required packages:
   ```bash
   pip install torch torchvision opencv-python matplotlib tqdm pillow
   ```

## Usage
1. Run the Human_Pose_Estimation.ipynb file
2. Ensure the saved model goes into the project directory
3. Extract_frames.py extracts frames from a video fileand saves them in a specified folder
4. Draw Keypoints happens from utils.py to draw keypoints and connections on the frames
5. main.py processes the video frames, performs pose estimation, and generates the output video

## Example: 

## Acknowledgements
This project utilizes the keypoint R-CNN model from torchvision and various open-source libraries
