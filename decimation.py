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

# decimation filter object
decimation2= rs.decimation_filter()

decimation2 = rs.decimation_filter()
decimation2.set_option(rs.option.filter_magnitude, 2)

decimation3 = rs.decimation_filter()
decimation3.set_option(rs.option.filter_magnitude, 3)

decimation4 = rs.decimation_filter()
decimation4.set_option(rs.option.filter_magnitude, 4)

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
    image_size_bytes = depth_color_image.nbytes  # Rozmiar w bajtach (wielkość tablicy NumPy)
    print(f"Rozmiar obrazu 0 {image_size_bytes} bajtów")

    # decymacja
    decimated_depth = decimation2.process(depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
    cv2.imshow("Decimal2-filter", colorized_depth)
    image_size_bytes = colorized_depth.nbytes  # Rozmiar w bajtach (wielkość tablicy NumPy)
    print(f"Rozmiar obrazu 1 {image_size_bytes} bajtów")

    decimated_depth = decimation3.process(depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
    cv2.imshow("Decimal3-filter", colorized_depth)
    image_size_bytes = colorized_depth.nbytes  # Rozmiar w bajtach (wielkość tablicy NumPy)
    print(f"Rozmiar obrazu 2 {image_size_bytes} bajtów")

    decimated_depth = decimation4.process(depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
    cv2.imshow("Decimal4-filter", colorized_depth)
    image_size_bytes = colorized_depth.nbytes  # Rozmiar w bajtach (wielkość tablicy NumPy)
    print(f"Rozmiar obrazu 3 {image_size_bytes} bajtów")

    # show frames
    cv2.imshow('Realsense-color', color_image)
    cv2.imshow("Realsense-depth", depth_color_image)
    
    # stop
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()