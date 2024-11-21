import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pyrealsense2 as rs

def preprocess_image(img):
    #img = Image.open(image_path).convert("RGB")
    #img = imgage.resize((640, 640))  # Resize to the model's expected input size
    img = np.array(img).astype(np.float32) / 255.0  # Convert to a NumPy array and normalize
    img = np.transpose(img, (2, 0, 1))  # Transpose the image to (channels, height, width)
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return torch.tensor(img)

#file = "runs/detect/yolov8n_custom6/weights/best.pt"
file = "yolov8s.pt"
model = YOLO(file)

colorizer = rs.colorizer()

# init realsense
pipeline = rs.pipeline()
camera_aconfig = rs.config()
camera_aconfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
camera_aconfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(camera_aconfig)
# init realsense

# read instrinctincs
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

while True:

    # get realsense frames
    frames = pipeline.wait_for_frames()

    # get aligned frames
    align_to = rs.stream.color
    align = rs.align(align_to)

    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_color_frame = colorizer.colorize(depth_frame)
    depth_color_image = np.asanyarray(depth_color_frame.get_data())

    # yolo detection from frame
    img_analyze = preprocess_image(color_image)
    res = model(img_analyze)
    boxes = res[0].boxes.xyxy.cpu().numpy()  
    confidences = res[0].boxes.conf.cpu().numpy()  
    class_ids = res[0].boxes.cls.cpu().numpy()  

    for i, box in enumerate(boxes):
        if model.names[int(class_ids[i])] == "bottle" and confidences[i] > 0.6:
            x1, y1, x2, y2 = map(int, box)

            label = f'{model.names[int(class_ids[i])]} {confidences[i]:.2f}'

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            dx = int((x1 + x2) // 2) # // - dzielenie calkowite
            dy = int((y1 + y2) // 2)

            cv2.rectangle(depth_color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(depth_color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # depth

    cv2.imshow('Realsense-color', color_image)
    cv2.imshow("Realsense-depth", depth_color_image)
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()