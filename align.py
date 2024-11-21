import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pyrealsense2 as rs

pipeline = rs.pipeline()
camera_aconfig = rs.config()
camera_aconfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
camera_aconfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(camera_aconfig)

colorizer = rs.colorizer()

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

    # show frames
    cv2.imshow('Realsense-color', color_image)
    cv2.imshow("Realsense-depth", depth_color_image)

    # stop
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()