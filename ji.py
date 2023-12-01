import random
import json
from ultralytics import YOLO
import subprocess
import os
import onnx
import onnxruntime as ort

model_class_names = ["no_u_turn",
"u_turn_allowed",
"bus_lane",
"motor_vehicle_lane",
"motor_vehicle_movement",
"combined_u_turn_and_left_turn_lane",
"u_turn_lane",
"combined_through_and_right_turn_lane",
"combined_through_and_left_turn_lane",
"through_lane",
"left_turn_lane",
"right_turn_lane",
"pedestrian_crossing",
"roundabout_traffic",
"drive_on_the_left_side_of_the_median_strip",
"drive_on_the_right_side_of_the_median_strip",
"turn_left_or_right",
"through_and_right_turn",
"through_and_left_turn",
"right_turn",
"left_turn",
"straight_ahead",
"restricted_parking_area_lifted",
"no_parking_area",
"long_term_parking_restriction_lifted",
"no_long_term_parking",
"no_parking",
"overtaking_allowed",
"no_overtaking",
"no_through_and_right_turn",
"no_through_and_left_turn",
"no_left_and_right_turn",
"no_through",
" no_right_turn",
"no_left_turn",
"no_entry_for_freight_vehicles",
"no_entry_for_motor_vehicles",
"no_entry",
"no_passage",
"yield_to_oncoming_traffic",
"slow_down_and_yield",
"stop_and_yield",
"no_entry_for_vehicles_transporting_dangerous_goods",
"no_entry_for_large_passenger_vehicles",
"other"
]

class YoloDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # path_without_extension, extension = os.path.splitext(model_path)
        # new_model_path = path_without_extension + '.engine'
        # self.model = YOLO(new_model_path)

    def detect(self, input_image):
        detections = self.model.predict(input_image, device='cuda',imgsz=(384,640),half=True)
        return detections

def init():
    # Set environment variables
    # os.environ['LD_LIBRARY_PATH'] = '/opt/TensorRT-8.0.1.6/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    # os.environ['LD_LIBRARY_PATH'] = '/opt/conda/lib/python3.7/site-packages/tensorrt:' + os.environ.get('LD_LIBRARY_PATH', '')
    # os.environ['PATH'] = '/usr/local/cuda-11.3/bin:' + os.environ.get('PATH', '')
    model = YoloDetector('/project/train/models/yolov8m/train3/weights/best.engine')
    return model

def process_image(net, input_image, args=None):
    detections = net.detect(input_image)   # list of 1 Results object

    # 处理检测结果
    target_info = []
    objects = []

    detections = detections[0]

    for detection in detections.boxes.data:
        target_info.append({
            "x": int(detection[0]),
            "y": int(detection[1]),
            "width": int(detection[2] - detection[0]),
            "height": int(detection[3] - detection[1]),
            "name": model_class_names[int(detection[5])],  # 使用模型类别名称
            "confidence": float(detection[4])
        })

        objects.append({
            "x": int(detection[0]),
            "y": int(detection[1]),
            "width": int(detection[2] - detection[0]),
            "height": int(detection[3] - detection[1]),
            "name": model_class_names[int(detection[5])],  # 使用模型类别名称
            "confidence": float(detection[4])
        })

    target_count = len(target_info)
    is_alert = True if target_count > 0 else False
    # print(target_info)

    return json.dumps({
        'algorithm_data': {
            'is_alert': is_alert,
            'target_count': target_count,
            'target_info': target_info
        },
        'model_data': {
            "objects": objects
        }
    })

model=init()
process_image(model, '/home/images/val/ZDStraffic_sign20231108_V4_train_street_892_116.jpg')