from flask import Flask, request, render_template, send_file
import os
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms as transforms
import utils
import cv2
import numpy as np

app = Flask(__name__)

# Load the model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, num_keypoints=17)
model.load_state_dict(torch.load('model.pth'))
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    if file:
        # Save the uploaded file
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Process the image
        image = Image.open(filepath).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)

        # Convert to numpy and draw keypoints
        frame = Image.open(filepath).convert("RGB")
        frame = np.array(frame)
        output_image = utils.draw_keypoints(outputs, frame)

        # Save the result
        result_filepath = os.path.join('results', file.filename)
        cv2.imwrite(result_filepath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        return send_file(result_filepath, mimetype='image/jpeg')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('results'):
        os.makedirs('results')
    app.run(debug=True)
