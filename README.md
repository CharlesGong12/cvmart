## 极市交通标志检测
### Baseline
使用yolov8作为基座进行改进，使用TensorRT进行推理加速。
### 改进
* 损失函数：
    * 通过F1曲线发现有些种类效果好，有些种类效果不好，因此使用focal loss。focal loss根据不同类别分类的难度进行加权。
    ![F1](./pics/f1.jpg)
    * 有些标志比较远、比较小，较远的标志需要着重学习。通过将iou和dfl损失函数除以bbox的面积，量化物体-摄像头的距离，赋予远、小的标志更大的权重。改动见 `loss.py` line 63.
    ![signs](./pics/signs.jpg)
    * model.train的参数dfl，[distribution focal loss](https://proceedings.neurips.cc/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf)，见 `train.py` line 56.
* 训练策略：
    * flip_lr会影响左右转弯等标识，故取消。
    * 通过训练日志发现没有过拟合现象，因此dropout和weight_decay都不用太大。增加难度比较大的数据增强，例如mixup。
    ![loss](./pics/loss.jpg)
    ![mixup](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*uYy0ru1y3H6ky3X7U3BTqA.png){ width=60% height=auto }
* 优化器
    * sam optimizer，[Sharpness-Aware Minimization for Efficiently Improving Generalization](https://github.com/davda54/sam)。论文中效果不错，在`trainer.py`实现，line 712, 473, 345, 261, 223. 目前代码中sam只支持用SGD，因为发现Adam会不收敛，loss很大。对超参数比较敏感，$\rho$和lr。务必截断梯度。注意自动混合精度。
    ![sam](./pics/sam.png)
    * adaptive = True是ASAM，sam的改进版。

* 没有尝试的方法：
    * 可以将数据集在split的时候分类，根据难度分成hard, easy，对不同难度的数据集分别训练，此方法尚未尝试。
    * 冻结训练效果不佳。
    * 在backbone中加skip-connection.