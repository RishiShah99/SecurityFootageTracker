import cv2
import os
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms as transforms
import utils
from tqdm import tqdm
import numpy as np

def extract_and_process_frames(video_file, output_video):
    # Load the model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, num_keypoints=17)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Set Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Process frames with progress bar
    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        orig_numpy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        orig_numpy = np.array(orig_numpy, dtype=np.float32) / 255
        image = Image.fromarray((orig_numpy * 255).astype(np.uint8)).convert('RGB')

        image = transform(image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)

        output_image = utils.draw_keypoints(outputs, frame)
        out.write(output_image)

    cap.release()
    out.release()

# Example usage
video_file = "test.mp4"
output_video = "output_video.mp4"
extract_and_process_frames(video_file, output_video)