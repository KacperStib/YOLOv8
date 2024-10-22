import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# def preprocess_image(image_path):
#     img = cv2.imread(image_path)  # Load image
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     img = img.transpose(2, 0, 1)  # Change data layout from HWC to CHW
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     img = torch.from_numpy(img).float() / 255.0  # Convert to tensor and normalize
#     return img

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # Resize to the model's expected input size
    img = np.array(img).astype(np.float32) / 255.0  # Convert to a NumPy array and normalize
    img = np.transpose(img, (2, 0, 1))  # Transpose the image to (channels, height, width)
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return torch.tensor(img)

file = "runs/detect/yolov8n_custom5/weights/best.pt"
model = YOLO(file)
model.info()
image_path = "test.jpg"

#input_image = preprocess_image(image_path)

output = model(image_path)