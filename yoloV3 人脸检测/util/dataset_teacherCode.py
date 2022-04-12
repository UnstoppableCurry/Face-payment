import glob
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from utils import letterbox, random_affine, xywh2xyxy, xyxy2xywh
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from util.utils import *
import os   #引用OS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# import utils.letterbox as letterbox
# import utils.random_affine as random_affine
# import utils.xyxy2xywh as xyxy2xywh

# 获取数据：图像数据和标签数据
class LoadImagesAndLabels(Dataset):
    # 2.1 初始化处理
    def __init__(self, path, batch_size, img_size=416, augment=False, multi_scale=False, root_path=os.path.curdir):
        # 获取图像文件
        with open(path, 'r') as file:
            img_files = file.read().splitlines()
            img_files = list(filter(lambda x: len(x) > 0, img_files))
        np.random.shuffle(img_files)
        self.img_files = img_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.multi_scale = multi_scale
        self.root_path = root_path
        self.scale_index = 0
        if self.multi_scale:
            self.img_size = img_size
        # 标签文件
        self.label_files = [x.replace('images', 'labels').replace('.jpg', '.txt') for x in self.img_files]

    # 2.2 数据量
    def __len__(self):
        return len(self.img_files)

    # 2.3 图像读取与增强
    def __getitem__(self, index):
        # 第一步：多尺度训练
        if self.multi_scale and (self.scale_index % self.batch_size == 0) and self.scale_index != 0:
            self.img_size = random.choice(range(11, 19)) * 32
        if self.multi_scale:
            self.scale_index += 1
        # 第二步：图像读取
        img_path = os.path.join(self.root_path, self.img_files[index][2:])
        img = cv2.imread(img_path)
        # 第三步：颜色增强
        augmnet_hsv = random.random() < 0.5
        if self.augment and augmnet_hsv:
            fraction = 0.5
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, None, 255, out=S)
            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, None, 255, out=V)
            img_hsv[:, :, 1] = S
            img_hsv[:, :, 2] = V
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        # 第四步：图像尺寸调整
        # 获取图像宽高
        h, w, _ = img.shape
        # resize+pad
        img, ratio, padw, padh = letterbox(img, height=self.img_size, augment=self.augment)
        # 获取图像标签
        label_path = os.path.join(self.root_path, self.label_files[index][2:])
        labels = []
        # 读取标签文件
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()
            x = np.array([x.split() for x in lines], dtype=np.float32)
            if x.size > 0:
                labels = x.copy()
                # 修改目标框的位置信息
                labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 1] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh

        # 第五步：几何变换的增强并调整label值
        # 仿射变化
        if self.augment:
            img, labels = random_affine(img, labels, degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1))
        # xywh
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) / self.img_size
        # 翻转
        # 水平翻转
        if self.augment:
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        # 第六步：获取图像和标注信息结果
        # 标注信息
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        # 图像处理
        # 1.通道
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # 2.类型
        img = np.ascontiguousarray(img, dtype=np.float32)
        # 3.归一化
        img /= 255.0
        return torch.from_numpy(img), labels_out, img_path, (h, w)

    # 2.4 获取batch数据
    @staticmethod
    def collate_fn(batch):
        img, label, img_path, hw = list(zip(*batch))
        for i, l in enumerate(label):
            l[:, 0] = i
        print([x[0].shape for x in batch])
        return torch.stack(img, 0), torch.cat(label, 0), img_path, hw
# 3 数据获取测试
if __name__ == "__main__":
    # 参数
    train_path = "/Users/yaoxiaoying/Desktop/人脸支付/02.code/datasets/yolo_widerface_open_train/anno/train.txt"
    root_path = "/Users/yaoxiaoying/Desktop/人脸支付/02.code/datasets"
    batch_size = 2
    img_size = 416
    num_workers = 2
    # 创建数据对象
    dataset = LoadImagesAndLabels(train_path, batch_size, img_size=img_size, augment=False, multi_scale=False,
                                  root_path=root_path)
    print(dataset.__len__())
    # dataloader来获取数据
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                            collate_fn=dataset.collate_fn)
    # 遍历loader
    for i, (imgs, targets, img_path_, _) in enumerate(dataloader):
        # 标注信息
        print('标注信息', targets)
        # 遍历imgs获取每一副图像进行展示
        for j in range(batch_size):
            # 对图像进行处理：反归一化，表示形式，通道，类型
            img_tmp = np.uint8(imgs[j].permute(1, 2, 0) * 255.0)[:, :, ::-1]
            # 显示
            cv2.imshow('result', img_tmp)
            cv2.waitKey(0)
            # 保存
            out_path = os.path.join("/Users/yaoxiaoying/Desktop/人脸支付/03.课堂代码/yolo_v3/result_aug",
                                    os.path.basename(img_path_[j]))
            cv2.imwrite(out_path,img_tmp)
        cv2.destroyAllWindows()
        continue
