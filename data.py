import re
with open("/project/train/src_repo/categories.txt", "r") as f:
    cats = f.readlines()

cats2id = dict()
id_ = 0
for cat in cats:
    if cat == "\n":
        continue
    pattern = re.compile(r'([\u4e00-\u9fa5]+|[a-zA-Z_]+)')
    matches = pattern.findall(cat)
    cats2id[matches[1]] = id_
    id_ += 1
id2cats = {k: v for v, k in cats2id.items()}

import os
import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET

roo = "/home/data"
os.makedirs("/home/labels", exist_ok=True)

# 解析XML文件
for dir in os.listdir(roo):
    for file in os.listdir(osp.join(roo, dir)):
        if file.endswith(".xml"):
            tree = ET.parse(osp.join(roo, dir, file))
            root = tree.getroot()

            filename = root.find('filename').text
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)

            s = ""
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                xmin = float(obj.find('bndbox/xmin').text)
                ymin = float(obj.find('bndbox/ymin').text)
                xmax = float(obj.find('bndbox/xmax').text)
                ymax = float(obj.find('bndbox/ymax').text)
                
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32).ravel()
                box[[0, 2]] /= width  # normalize x by width
                box[[1, 3]] /= height  # normalize y by height
                box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # xywh
                s += '%g %.6f %.6f %.6f %.6f\n' % (cats2id[name], *box)
            with open("/home/labels/"+file.split(".")[0]+".txt", "w+") as f:
                f.write(s)