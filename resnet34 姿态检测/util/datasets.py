import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import json


# from data_agu import *


class LoadImagesAndLabels(Dataset):
    def __init__(self, ops, img_size=(224, 224), flag_agu=False):
        '''
        初始化函数
        :param ops:配置文件读取对象
        :param img_size:图像输入尺寸
        :param flag_agu:工具类..暂定
        '''
        file_list = []
        bboxes_list = []
        angles_list = []
        # 计数
        idx = 0
        # 获取图像路径
        images_path = ops.train_path + 'images/'
        for f_ in os.listdir(images_path):
            print('加载数据进度-->', idx*100 / len(os.listdir(images_path)))
            # 获取对应label路径
            label_path = (images_path + f_).replace("images", "labels").replace(".jpg", ".json")
            # 读取json
            f = open(label_path, encoding='utf8')
            dict = json.load(f)
            f.close()
            # 获取角度和bbox目标框
            angle = dict["euler_angle"]
            bbox = dict['bbox']
            idx += 1
            print("  images : {}".format(idx), end="\r")
            file_list.append(images_path + f_)
            bboxes_list.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            angles_list.append((angle['yaw'], angle['pitch'], angle['roll']))
        self.files = file_list
        self.bboxes = bboxes_list
        self.angles = angles_list
        self.img_size = img_size
        self.flag_agu = flag_agu

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        yaw, pitch, roll = self.angles[index]
        bbox = self.bboxes[index]
        # 去读图像
        img = cv2.imread(img_path)
        # 获取人脸范围
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        face_w = xmax - xmin
        face_h = ymax - ymin
        # 随机扩展
        x_min = int(xmin - random.randint(-6, int(face_w * 3 / 5)))
        y_min = int(ymin - random.randint(-6, int(face_h * 2 / 3)))
        x_max = int(xmax + random.randint(-6, int(face_w * 3 / 5)))
        y_max = int(ymax + random.randint(-12, int(face_h * 2 / 5)))
        # clip
        x_min = np.clip(x_min, 0, img.shape[1] - 1)
        x_max = np.clip(x_max, 0, img.shape[1] - 1)
        y_min = np.clip(y_min, 0, img.shape[0] - 1)
        y_max = np.clip(y_max, 0, img.shape[0] - 1)
        try:
            face_crop = img[y_min:y_max, x_min:x_max, :]
        except:
            face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        if random.random() >= 0.5:
            face_crop = cv2.flip(face_crop, 1)
            yaw = -yaw
            roll = -roll  # 俯仰角不变 眼镜蛇机动~
        img_ = cv2.resize(face_crop, self.img_size, interpolation=random.randint(0, 4))
        # 颜色增强
        if self.flag_agu:
            if random.random() > 0.7:
                img_hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
                # 随机生成增强的数值
                hue_x = random.randint(-10, 10)
                #  对H通道进行增强，并对范围进行调整，0 ~180
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_x)
                img_hsv[:, :, 0] = np.maximum(img_hsv[:, :, 0], 0)
                img_hsv[:, :, 0] = np.minimum(img_hsv[:, :, 0], 180)
                img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        # 数据归一化
        img_ = img_.astype(np.float32)[:, :, ::-1]  # 倒装
        # 归一化
        img_ = (img_ - 128.) / 256.
        img_ = img_.transpose(2, 0, 1)  # HWC-->CHW 增加计算效率
        # 角度归一化
        yaw = yaw / 90.
        pitch = pitch / 90.
        roll = roll / 90.
        # 构成一个一维数组
        angles_ = np.array([yaw, pitch, roll]).ravel()
        return img_, angles_


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=' Project Face Euler Angle Train')
    #  yaw,pitch,roll
    parser.add_argument('--num_classes', type=int, default=3,
                        help='num_classes')
    # 训练集标注信息
    parser.add_argument('--train_path', type=str,
                        default='/root/cv/dataset/人脸/datasets/face_euler_angle_datasets_mini/', help='train_path')
    # 训练每批次图像数量
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    # 训练线程数
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers')
    # 输入模型图片尺寸
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')
    # 是否进行数据增强
    parser.add_argument('--flag_agu', type=bool, default=True,
                        help='data_augmentation')

    import matplotlib.pyplot as plt

    # --------------------------------------------------------------------------
    ops = parser.parse_args()  # 解析添加参数
    # 数据加载
    dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu)
    print('len train datasets : %s' % (dataset.__len__()))
    # Dataloader获取batchsize的数据
    dataloader = DataLoader(dataset,
                            batch_size=ops.batch_size,
                            num_workers=ops.num_workers,
                            shuffle=True)
    # 获取每个batch的训练数据
    for i, (imgs_, angles_) in enumerate(dataloader):
        # 打印角度信息
        print(angles_)
        for j in range(ops.batch_size):
            # 结果展示:反归一化，表示形式CHW->HWC,类型转换，RGB->BGR
            # cv2.imshow('result',np.uint8(imgs_[i].permute(1, 2, 0)*256.0+128.0)[:,:,::-1])
            # cv2.waitKey(0)
            if imgs_.shape[0]==1:
                plt.imshow(np.uint8(imgs_[0].permute(1, 2, 0) * 256.0 + 128.0)[:, :, ::-1])
            else:
                plt.imshow(np.uint8(imgs_[i].permute(1, 2, 0) * 256.0 + 128.0)[:, :, ::-1])

            plt.show()

    cv2.destroyAllWindows()
