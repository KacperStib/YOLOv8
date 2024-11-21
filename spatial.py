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

spatialM = rs.spatial_filter()
spatialM.set_option(rs.option.filter_magnitude, 5)  # Siła wygładzania

spatialA = rs.spatial_filter()
spatialA.set_option(rs.option.filter_smooth_alpha, 0.25)  # Współczynnik wygładzania

spatialD = rs.spatial_filter()
spatialD.set_option(rs.option.filter_smooth_delta, 50)  # Zachowanie krawędzi

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

    # filtr przestrzenny
    filtered_depth = spatial.process(depth_frame)
    colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    cv2.imshow("Spatial-filter", colorized_depth)

    filtered_depth = spatialM.process(depth_frame)
    colorized_depth1 = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    cv2.imshow("Spatial-filterM", colorized_depth1)

    filtered_depth = spatialA.process(depth_frame)
    colorized_depth2 = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    cv2.imshow("Spatial-filterA", colorized_depth2)

    filtered_depth = spatialD.process(depth_frame)
    colorized_depth3 = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    cv2.imshow("Spatial-filterD", colorized_depth3)

    # show frames
    cv2.imshow('Realsense-color', color_image)
    cv2.imshow("Realsense-depth", depth_color_image)
    
    # stop
    if cv2.waitKey(1) == ord('q'):
        # cv2.imwrite('spatial-normal.png', depth_color_image)
        # cv2.imwrite('spatialKozak.png', colorized_depth)
        # cv2.imwrite('spatialM.png', colorized_depth1)
        # cv2.imwrite('spatialA.png', colorized_depth2)
        # cv2.imwrite('spatialD.png', colorized_depth3)

        # psnr_value = cv2.PSNR(depth_color_image, colorized_depth)
        # print(f"PSNR1: {psnr_value} dB")
        # psnr_value = cv2.PSNR(depth_color_image, colorized_depth1)
        # print(f"PSNR2: {psnr_value} dB")
        # psnr_value = cv2.PSNR(depth_color_image, colorized_depth2)
        # print(f"PSNR3: {psnr_value} dB")
        # psnr_value = cv2.PSNR(depth_color_image, colorized_depth3)
        # print(f"PSNR4: {psnr_value} dB")

        break

pipeline.stop()
cv2.destroyAllWindows()