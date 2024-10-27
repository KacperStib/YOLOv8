import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pyrealsense2 as rs

def preprocess_image(img):
    #img = Image.open(image_path).convert("RGB")
    #img = img.resize((640, 640))  # Resize to the model's expected input size
    img = np.array(img).astype(np.float32) / 255.0  # Convert to a NumPy array and normalize
    img = np.transpose(img, (2, 0, 1))  # Transpose the image to (channels, height, width)
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return torch.tensor(img)

file = "runs/detect/yolov8n_custom6/weights/best.pt"
model = YOLO(file)

# init realsense
pipeline = rs.pipeline()
camera_aconfig = rs.config()
camera_aconfig.enable_stream(rs.stream.depth, 640, 640, rs.format.z16, 30)
camera_aconfig.enable_stream(rs.stream.color, 640, 640, rs.format.bgr8, 30)
pipeline.start(camera_aconfig)
# init realsense

while True:

    # get realsense frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    #if not depth_frame or not color_frame:
        #return None, None
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # get realsense frames

    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
	#filled_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(filled_depth_image, alpha=0.03), cv2.COLORMAP_JET)
	# color_seg = color_image.copy()
	# depth_seg = depth_colormap.copy()
	# filled_depth_seg = filled_depth_colormap.copy()

    img_analyze = preprocess_image(color_image)
    res = model(img_analyze)
    boxes = res[0].boxes.xyxy.cpu().numpy()  
    confidences = res[0].boxes.conf.cpu().numpy()  
    class_ids = res[0].boxes.cls.cpu().numpy()  

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        label = f'{model.names[int(class_ids[i])]} {confidences[i]:.2f}'

        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        #depth
        dx = int((x1 + x2) // 2) #// dzielenie calkowite
        dy = int((y1 + y2) // 2)
        depth = depth_frame.get_distance(dy, dx)

        label = f'{label} {depth:.2f}'
        cv2.rectangle(depth_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(depth_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #depth

    cv2.imshow('Realsense-color', color_imgage)
    cv2.imshow("Realsense-depth", depth_image)
    if cv2.waitKey(1) == ord('q'):
        break


pipeline.stop()
cv2.destroyAllWindows()
