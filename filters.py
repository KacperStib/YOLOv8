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

# spatial filter object
spatial = rs.spatial_filter()

spatialM = rs.spatial_filter()
spatialM.set_option(rs.option.filter_magnitude, 5)  # Siła wygładzania

spatialA = rs.spatial_filter()
spatialA.set_option(rs.option.filter_smooth_alpha, 1)  # Współczynnik wygładzania

spatialD = rs.spatial_filter()
spatialD.set_option(rs.option.filter_smooth_delta, 50)  # Zachowanie krawędzi

# temporal filter object
temporal = rs.temporal_filter()

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
    # image_size_bytes = depth_color_image.nbytes  # Rozmiar w bajtach (wielkość tablicy NumPy)
    # print(f"Rozmiar obrazu 0 {image_size_bytes} bajtów")

    # decimal_filter
    # decimated_depth = decimation2.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
    # cv2.imshow("Decimal2-filter", colorized_depth)
    # image_size_bytes = colorized_depth.nbytes  # Rozmiar w bajtach (wielkość tablicy NumPy)
    # print(f"Rozmiar obrazu 1 {image_size_bytes} bajtów")

    # decimated_depth = decimation3.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
    # cv2.imshow("Decimal3-filter", colorized_depth)
    # image_size_bytes = colorized_depth.nbytes  # Rozmiar w bajtach (wielkość tablicy NumPy)
    # print(f"Rozmiar obrazu 2 {image_size_bytes} bajtów")

    # decimated_depth = decimation4.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
    # cv2.imshow("Decimal4-filter", colorized_depth)
    # image_size_bytes = colorized_depth.nbytes  # Rozmiar w bajtach (wielkość tablicy NumPy)
    # print(f"Rozmiar obrazu 3 {image_size_bytes} bajtów")

    # spatial filter
    # filtered_depth = spatialM.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    # cv2.imshow("Spatial-filterM", colorized_depth)

    # filtered_depth = spatialA.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    # cv2.imshow("Spatial-filterA", colorized_depth)

    # filtered_depth = spatialD.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
    # cv2.imshow("Spatial-filterD", colorized_depth)

    # temporal
    # frames = []
    # start_time = time.time()
    # for x in range(10):
    #     frameset = pipeline.wait_for_frames()
    #     frames.append(frameset.get_depth_frame())
    # for x in range(10):
    #     temp_filtered = temporal.process(frames[x])
    # colorized_depth3 = np.asanyarray(colorizer.colorize(temp_filtered).get_data())
    # cv2.imshow("Temporal-filter", colorized_depth3)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Czas wykonania: {execution_time} sekund")

    # start_time = time.time()
    # for x in range(3):
    #     frameset = pipeline.wait_for_frames()
    #     frames.append(frameset.get_depth_frame())
    # for x in range(3):
    #     temp_filtered = temporal.process(frames[x])
    # colorized_depth1 = np.asanyarray(colorizer.colorize(temp_filtered).get_data())
    # cv2.imshow("Temporal-filter3", colorized_depth1)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Czas wykonania: {execution_time} sekund")

    # start_time = time.time()
    # for x in range(5):
    #     frameset = pipeline.wait_for_frames()
    #     frames.append(frameset.get_depth_frame())
    # for x in range(5):
    #     temp_filtered = temporal.process(frames[x])
    # colorized_depth2 = np.asanyarray(colorizer.colorize(temp_filtered).get_data())
    # cv2.imshow("Temporal-filter5", colorized_depth2)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Czas wykonania: {execution_time} sekund")

    
    # hole filling
    # filled_depth = hole_filling0.process(depth_frame)
    # colorized_depth1 = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    # cv2.imshow("Hole-filling-filter1", colorized_depth1)

    # filled_depth = hole_filling1.process(depth_frame)
    # colorized_depth2 = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    # cv2.imshow("Hole-filling-filter2", colorized_depth2)

    # filled_depth = hole_filling2.process(depth_frame)
    # colorized_depth3 = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    # cv2.imshow("Hole-filling-filte3", colorized_depth3)

    # together
    # filtered_depth = spatial.process(depth_frame)
    # filled_depth = hole_filling.process(depth_frame)
    # colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
    # cv2.imshow("Spatial-Hole-filling-filter", colorized_depth)

    # show frames
    cv2.imshow('Realsense-color', color_image)
    cv2.imshow("Realsense-depth", depth_color_image)
    
    # stop
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()