from ultralytics import YOLO
# TODO: run/
model = YOLO('/project/train/models/yolov8m/train3/weights/best.pt')       
# model.export(format='torchscript', device='cuda:0', imgsz=(384,640))
model.export(format='engine', device='cuda:0', imgsz=(384,640), half=True)
