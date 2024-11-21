from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8s.pt')
 
# Training.
results = model.train(
   data='data.yaml',
   imgsz=640,
   epochs=100,
   batch=8,
   name='yolov8s_custom'
)
