import cv2
import torch
import torchvision
from torchvision import transforms

# Load the pre-trained SSD model from torchvision
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()

# COCO dataset classes
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Open video
cap = cv2.VideoCapture("Video_3.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

# Define the transform to convert the frames to tensors
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loop through the video frames
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the transformation
        input_frame = transform(frame).unsqueeze(0)

        detections = model(input_frame)[0]

        for i, score in enumerate(detections['scores']):
            if score > 0.5:
                box = detections['boxes'][i].cpu().numpy().astype(int)
                label = COCO_CLASSES[detections['labels'][i]]
                if label == "person":
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and video writer
cap.release()
out.release()
cv2.destroyAllWindows()
