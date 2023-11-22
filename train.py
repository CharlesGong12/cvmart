from ultralytics import YOLO
import os
import shutil
os.environ['WANDB_DISABLED'] = 'true'

model = YOLO('yolov8m.yaml').load('/project/train/models/yolov8m/train/weights/last.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/project/train/src_repo/yolov8m_conf.yaml',
 epochs=20, 
 imgsz=640, 
 device='cuda:0',
 batch=-1, 
 optimizer='AdamW',
 lr0=1e-4,
 warmup_epochs=2,
 dropout=0.1,
 val=True,
 weight_decay=1e-5,
 resume=True,
 project='/project/train/models/yolov8m',   #save path
 fliplr=0,
 copy_paste=0.2,
 degrees=3,
 patience=5,
 exist_ok=True
 )

model.export(format='engine', device='cuda')

# 找到具有最大后缀数字的文件夹
# detect_folder = '/project/runs/detect'
# target_path = '/project/train/models/yolov8m'
# train_folders = [folder for folder in os.listdir(detect_folder) if folder.startswith('train')]

# if train_folders:
#     if len(train_folders)==1:
#         max_train_folder='train'
#     else:
#         max_suffix = max([int(folder.split('train')[-1]) for folder in train_folders[1:]])
#         max_train_folder = f'train{max_suffix}'

#     # 复制 best.pt 文件到目标文件夹
#     source_best_pt = os.path.join(detect_folder, max_train_folder, 'weights', 'best.pt')
#     print(f'ckpt found from: {source_best_pt}.')
#     target_best_pt = os.path.join(target_path, 'best.pt')
#     os.makedirs(target_path, exist_ok=True)
#     print(f'target path made: {target_path}.')

#     shutil.copyfile(source_best_pt, target_best_pt)
#     print(f"Copied best.pt from {max_train_folder} to {target_best_pt}")
# else:
#     print(f"No 'train' folders found in {detect_folder}")
