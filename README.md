# Security Footage Human Detection

This project aims to detect humans in security footage using a custom-trained Faster R-CNN model. The model is trained on extracted frames from a provided video and then used to process additional videos to detect humans and draw bounding boxes around them.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Extracting Frames](#extracting-frames)
  - [Labeling Frames](#labeling-frames)
  - [Training the Model](#training-the-model)
  - [Using the Trained Model](#using-the-trained-model)
- [Results](#results)

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/SecurityFootageHumanDetection.git
   cd SecurityFootageHumanDetection

2. Install the required packages:
   ```sh
   pip install -r requirements.txt

## Usage
### Extracting Frames
1. Run the script to extract frames from a video:
   ```sh
   python extract_frames.py --video_path your_video.mp4 --output_folder output_frames
### Labeling Frames
1. Label the extracted frames using the LabelImg tool:
   ```sh
   Use the LabelImg to label the extracted frames. Save the annotations in Pascal VOC format.
### Training the model
1. Train the Faster R-CNN model using the labeled frames:
   ```sh
   Follow instructions in train-model.ipynb to train the model.
### Using the Trained Model
1. Download the trained model and place it in the project directory.
2. Run the script to process additional videos:
    ```sh
    python process_videos.py
## Results
The trained model is able to detect humans in security footage with high accuracy. The model is able to draw bounding boxes around humans in the video frames, allowing for easy identification and tracking of individuals.
```


