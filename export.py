from ultralytics import YOLO
model = YOLO('/project/train/models/yolov8m/train3/weights/last.pt')
model.export(format='engine', device='cuda')