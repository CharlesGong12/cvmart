#!/bin/bash

cp /project/train/src_repo/yolov8n.pt yolov8n.pt

python /project/train/src_repo/data.py
python /project/train/src_repo/split.py --mode train
python /project/train/src_repo/train.py
