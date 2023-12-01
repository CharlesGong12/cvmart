from ultralytics import YOLO
from utils import freeze_head, freeze_backbone
import os
import shutil
import argparse
import torch
os.environ['WANDB_DISABLED'] = 'true'

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--freeze_part', default=None, choices=['backbone', 'head'], help='Part to freeze')
parser.add_argument('--epochs', default=15, help='training epochs', type=int)
args = parser.parse_args()

# TODO
# build from YAML and transfer weights
if args.freeze_part is not None:
    if args.freeze_part == 'backbone':
        model = YOLO('/project/train/models/yolov8m/train3/weights/best.pt')
        model.add_callback("on_pretrain_routine_start", freeze_backbone)
    if args.freeze_part == 'head':
        model = YOLO('/project/train/models/yolov8m/train3/weights/best.pt')
        model.add_callback("on_pretrain_routine_start", freeze_head)
    results = model.train(data='/project/train/src_repo/yolov8m_conf.yaml',
    epochs=args.epochs,  # TODO
    imgsz=640, 
    device='cuda:0',
    batch=-1,   # TODO
    optimizer='AdamW',
    lr0=1e-4,
    warmup_epochs=0,
    dropout=0.1,
    val=True,
    weight_decay=1e-5,
    project='/project/train/models/yolov8m',   #save path
    fliplr=0,
    copy_paste=0.3,
    degrees=4,
    close_mosaic=0,
    perspective=1e-4,
    translate=0.05,
    scale=0.25,
    patience=5,
    exist_ok=True,
    name='train3'
    )
else:
    model = YOLO('/project/train/src_repo/yolov8m.pt')
            
    # Train the model
    results = model.train(data='/project/train/src_repo/yolov8m_conf.yaml',
    epochs=args.epochs,  # TODO
    imgsz=(384,640), 
    device='cuda:0',
    batch=-1,   # TODO
    optimizer='AdamW',
    dfl=2.5,
    cos_lr=True,
    lr0=1e-2,
    warmup_epochs=3,
    dropout=0.1,
    val=True,
    weight_decay=1e-6,
    project='/project/train/models/yolov8m',   #save path
    fliplr=0,
    close_mosaic=8,
    patience=5,
    mixup=0.3,
    exist_ok=True,
    name='train3',
    )
