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

# spatial filter object
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 0.25) 
spatial.set_option(rs.option.filter_smooth_delta, 50)

# temporal filter object
temporal = rs.temporal_filter()

# hole filling
hole_filling = rs.hole_filling_filter()

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

    # klatki
    frames = []

    # zbieranie klatek
    for x in range(3):
        frameset = pipeline.wait_for_frames()
        frames.append(frameset.get_depth_frame())

    # temporal
    for x in range(3):
        temp_filtered = temporal.process(frames[x])

    # spatial
    spatial_filtered = spatial.process(temp_filtered)

    # hole
    hole_filtered = hole_filling.process(spatial_filtered)
    colorized_depth = np.asanyarray(colorizer.colorize(hole_filtered).get_data())
    cv2.imshow("Hole-filling-filter1", colorized_depth)

    # show frames
    cv2.imshow('Realsense-color', color_image)
    cv2.imshow("Realsense-depth", depth_color_image)
    
    # stop
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite('final-color.png', color_image)
        cv2.imwrite('final-depth.png', depth_color_image)
        cv2.imwrite('final-filter.png', colorized_depth)
        break

pipeline.stop()
cv2.destroyAllWindows()