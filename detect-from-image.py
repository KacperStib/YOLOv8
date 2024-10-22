import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    #img = img.resize((224, 224))  # Resize to the model's expected input size
    img = np.array(img).astype(np.float32) / 255.0  # Convert to a NumPy array and normalize
    img = np.transpose(img, (2, 0, 1))  # Transpose the image to (channels, height, width)
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return torch.tensor(img)

file = "runs/detect/yolov8n_custom6/weights/best.pt"
model = YOLO(file)
model.info()
image_path = "test.jpg"

input_image = preprocess_image(image_path)

res = model(input_image)

print(res)

boxes = res[0].boxes.xyxy.cpu().numpy()  
confidences = res[0].boxes.conf.cpu().numpy()  
class_ids = res[0].boxes.cls.cpu().numpy()  

print(class_ids, confidences)

img = cv2.imread(image_path)

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)

    label = f'{model.names[int(class_ids[i])]} {confidences[i]:.2f}'

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Konwersja do RGB na potrzeby matplotlib
plt.axis('off')
plt.show()
