import os
import shutil
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Script to split data and labels into train/val and create file lists.')
parser.add_argument('--mode', default='train', choices=['debug', 'train'], help='Mode for script execution: debug or train')
args = parser.parse_args()

if args.mode =='train':
    data_folders = ['/home/data/2814', '/home/data/2868', '/home/data/2863']
else:
    data_folders = ['/home/data/2863']
labels_folder = '/home/labels/'
output_folder = '/home/'

# Write train and validation file lists
def write_file_list(image_files, split):
    with open(f'/home/{split}_list.txt', 'w') as f:  
        for file in image_files:
            label_file = file.replace('.jpg', '.txt')
            f.write(f'{output_folder}images/{split}/{file} {output_folder}labels/{split}/{label_file}\n')


os.makedirs(os.path.join(output_folder, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'labels/val'), exist_ok=True)

# all files gotten
for idx, data_folder in enumerate(data_folders):
    image_files = [file for file in os.listdir(data_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    for file in train_files:
        shutil.copyfile(os.path.join(data_folder, file), os.path.join(output_folder, 'images/train', file))
        label_file = file.replace('.jpg', '.txt')
        shutil.copyfile(os.path.join(labels_folder, label_file), os.path.join(output_folder, 'labels/train', label_file))

    for file in val_files:
        shutil.copyfile(os.path.join(data_folder, file), os.path.join(output_folder, 'images/val', file))
        label_file = file.replace('.jpg', '.txt')
        shutil.copyfile(os.path.join(labels_folder, label_file), os.path.join(output_folder, 'labels/val', label_file))


train_images = os.listdir(os.path.join(output_folder, 'images/train'))
val_images = os.listdir(os.path.join(output_folder, 'images/val'))

write_file_list(train_images, 'train')
write_file_list(val_images, 'val')

