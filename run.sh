#!/bin/bash
mv /project/train/models/yolov8m/train/val*pred.jpg /project/train/result-graphs/old_val.jpg

cp /project/train/src_repo/yolov8n.pt yolov8n.pt

python /project/train/src_repo/data.py
python /project/train/src_repo/split.py --mode train
python /project/train/src_repo/train.py

mv /project/train/models/yolov8m/train3/F1_curve.png /project/train/result-graphs

mv /project/train/models/yolov8m/train3/val*pred.jpg /project/train/result-graphs

