import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import transforms as transforms
import matplotlib.pyplot as plt
import utils

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

# Load image
image_path = 'frames/frame_1.jpg'
image = Image.open(image_path).convert('RGB')

# NumPy copy of the image for OpenCV functions
orig_numpy = np.array(image, dtype=np.float32)
orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255
image = transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)

output_image = utils.draw_keypoints(outputs, orig_numpy)

# visualize the image
cv2.imshow('Keypoint image', output_image)
cv2.waitKey(0)
