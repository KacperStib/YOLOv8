import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from cv_bridge import CvBridge


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Inicjalizowanie modelu YOLOv8
        self.model = YOLO("b1.pt")  # Ścieżka do modelu YOLO
        
        # Inicjalizacja kamery RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        
        # Parametry kamery
        self.intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # Publisher do wysyłania punktów 3D
        self.point_publisher = self.create_publisher(PointStamped, 'detected_object_coordinates', 10)
        
        # Konwersja obrazu ROS na OpenCV
        self.bridge = CvBridge()

        self.get_logger().info("Object Detection Node started.")

    def preprocess_image(self, img):
        img = np.array(img).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # Konwersja na kanały, wysokość, szerokość
        img = np.expand_dims(img, axis=0)  # Dodanie wymiaru wsadu
        return torch.tensor(img)

    def detect_objects(self, frame):
        # Preprocessing obrazu przed detekcją
        img_analyze = self.preprocess_image(frame)
        results = self.model(img_analyze)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # Współrzędne prostokątów
        confidences = results[0].boxes.conf.cpu().numpy()  # Pewność detekcji
        class_ids = results[0].boxes.cls.cpu().numpy()  # Identyfikatory klas obiektów
        return boxes, confidences, class_ids

    def process_frames(self):
        # Otrzymanie ramek z kamery
        frames = self.pipeline.wait_for_frames()

        align_to = rs.stream.color
        align = rs.align(align_to)
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_frame_color = rs.colorizer().colorize(depth_frame)
        depth_image = np.asanyarray(depth_frame_color.get_data())

        # Detekcja obiektów na obrazie
        boxes, confidences, class_ids = self.detect_objects(color_image)

        for i, box in enumerate(boxes):
            if confidences[i] > 0.4:
                x1, y1, x2, y2 = map(int, box)

                # Obliczanie współrzędnych środka obiektu
                dx = int((x1 + x2) // 2)
                dy = int((y1 + y2) // 2)

                # Pobieranie głębokości z punktu na obrazie
                depth = depth_frame.get_distance(dx, dy)

                if depth > 0:  # Jeśli głębokość jest dostępna
                    # Konwersja na współrzędne 3D
                    point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [dx, dy], depth)

                    # Tworzenie wiadomości ROS
                    point_msg = PointStamped()
                    point_msg.header = Header()
                    point_msg.header.stamp = self.get_clock().now().to_msg()
                    point_msg.header.frame_id = 'camera_link'

                    point_msg.point.x = point_3d[0]
                    point_msg.point.y = point_3d[1]
                    point_msg.point.z = point_3d[2]

                    # Publikowanie współrzędnych 3D
                    self.point_publisher.publish(point_msg)

                    # Rysowanie prostokąta i tekstu na obrazie
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{self.model.names[int(class_ids[i])]} {confidences[i]:.2f}'
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Wyświetlanie obrazu z detekcją
        cv2.imshow("Detected Objects", color_image)
        cv2.waitKey(1)

    def run(self):
        while rclpy.ok():
            self.process_frames()

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)

    node = ObjectDetectionNode()
    node.run()

    node.stop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
