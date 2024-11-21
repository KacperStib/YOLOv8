import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import time

pipeline = rs.pipeline()
camera_aconfig = rs.config()
camera_aconfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
camera_aconfig.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(camera_aconfig)

colorizer = rs.colorizer()

# hole filling
hole_filling0 = rs.hole_filling_filter()
hole_filling0.set_option(rs.option.holes_fill, 0)

hole_filling1 = rs.hole_filling_filter()
hole_filling1.set_option(rs.option.holes_fill, 1)

hole_filling2 = rs.hole_filling_filter()
hole_filling2.set_option(rs.option.holes_fill, 2)

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
    
    # hole filling
    filled_depth = hole_filling0.process(depth_frame)
    colorized_depth1 = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    cv2.imshow("Hole-filling-filter1", colorized_depth1)

    filled_depth = hole_filling1.process(depth_frame)
    colorized_depth2 = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    cv2.imshow("Hole-filling-filter2", colorized_depth2)

    filled_depth = hole_filling2.process(depth_frame)
    colorized_depth3 = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    cv2.imshow("Hole-filling-filte3", colorized_depth3)

    # show frames
    cv2.imshow('Realsense-color', color_image)
    cv2.imshow("Realsense-depth", depth_color_image)
    
    # stop
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()