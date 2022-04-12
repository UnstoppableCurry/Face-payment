import glob
import math
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import sys

# from utils import letterbox, random_affine, xywh2xyxy, xyxy2xywh

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from util.utils import *
import os  # 引用OS


class LoadImagesAndLabels(Dataset):
    # 初始化
    def __init__(self, path, batch_size, img_size=416, augment=False, multi_scale=False, root_path=os.path.curdir):
        '''
        :param path:
        :param batch_size:
        :param img_size:
        :param augment:
        :param multi_scale:
        :param root_path:
        '''
        with open(path, 'r',encoding='utf-8') as file:
            img_files = file.read().splitlines()
            img_files = list(filter(lambda x: len(x) > 0, img_files))
        # 读取所有图片文件
        np.random.shuffle(img_files)  # 乱序处理数据
        self.img_files = img_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.multi_scale = multi_scale
        self.root_path = root_path
        self.scale_index = 0
        if self.multi_scale:
            self.img_size = img_size
        self.label_file = [
            x.replace('images', 'labels').replace('./', '/root/cv/dataset/人脸/datasets/').replace('.jpg', '.txt') for x
            # x.replace('images', 'labels').replace('./', '/www/dataset/yolo_helmet_train/').replace('.jpg', '.txt') for x
            in self.img_files]

    def __len__(self):
        '''
        数据量
        :return:
        '''
        return len(self.img_files)

    def __getitem__(self, index):
        '''
        图像读取与数据增强
        :param index:
        :return:
        '''
        # 是否进行多尺度训练
        if self.multi_scale and (self.scale_index % self.batch_size == 0) and self.scale_index != 0:
            # batch必须能整除才行...
            self.img_size = random.choice(range(11, 19)) * 32
            # 尺寸从11-19  必须是32的倍数
        if self.multi_scale:
            self.scale_index += 1
        # 图像读取
        # img_path = os.path.join(self.img_files[index].replace('./', '/www/dataset/yolo_helmet_train/'))
        img_path = os.path.join(self.img_files[index].replace('./', '/root/cv/dataset/人脸/datasets/'))
        # print(img_path, '-------')
        img = cv2.imread(img_path)
        # print(img.shape)
        # print(img)
        # 颜色增强 在HSV色彩空间进行颜色的处理 色调(H)，饱和度(S)，明度(V) 不改变颜色的基础上更改颜色饱和度和亮度
        augment_hsv = random.random() < 0.5  # 去 0 - 1  相同概率出现 类似teacher-forching
        if self.augment and augment_hsv:
            fraction = 0.5
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            # 生成随机数a 在[0,5, 1.5]之间，对S通道进行处理
            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, None, 255, out=S)
                # 截取函数,将S这个数组中的元素截取到指定范围内,小于None的就是None,大于255的就取255,out参数是将截取后的数放入数组中,但是要保持数据shape一致
            # 生成随机数a 在[0,5, 1.5]之间，对通道进行处理
            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, None, 255, out=V)  # V尺度做相同操作
            # 赋值给原图像
            img_hsv[:, :, 1] = S
            img_hsv[:, :, 2] = V
            # 颜色空间转换为BGR，完成图像增强
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        # 将图像尺度调整为正方形,不能进行直接resize 会破坏图像的比例导致失真
        h, w, _ = img.shape
        # resize+pad填充
        img, ratio, padw, padh = letterbox(img, height=self.img_size, augment=self.augment)
        # 获取图像标签
        label_path = os.path.join(self.label_file[index])
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
        # 几何变换增强,并处理label值
        # 仿射变换
        if self.augment:
            img, labels = random_affine(img, labels, degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1))
        # xywh变换
        nl = len(labels)
        if nl:
            # 不为空就计算     转化xyxy为xywh，且并归一化
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) / self.img_size  # 第一个是图片index,第二个是cls 后面四个是xywh
        # 翻转
        if self.augment:
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]  # 水平翻转 x 变为1-x

        # 获取图像和标注信息结果
        # 标注
        labels_out = torch.zeros((nl, 6))  # 生成一个全零数据存储label值
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)  # 若标签不为空，将其填充在全零数组中

        # 图像处理
        # 通道更改
        img = img[:, :, ::-1].transpose(2, 0, 1)  # 通道倒序排列后更改位置
        # 通道BGR to RGB，表示形式转换为3x416x416（CHW） chw比hwc效率更高效果更好
        # 类型
        img = np.ascontiguousarray(img, dtype=np.float32)  # 更改数据类型
        # 内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        # 归一化
        img /= 255.0
        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod  # 静态方法
    def collate_fn(batch):
        '''
        静态方法不需要实例对象也可调用,这里是为了实现自定义的batch读取
        :param batch:
        :return:
        '''
        img, label, img_path, hw = list(zip(*batch))
        for i, l in enumerate(label):
            l[:, 0] = i  # 图片下标设置为1
        return torch.stack(img, 0), torch.cat(label, 0), img_path, hw


if __name__ == '__main__':
    # 测试
    # 指定文件路径
    # 数据路径

    root_path = '/root/cv/dataset/人脸/datasets'
    # txt文件的路径
    path = '/www/dataset/yolo_helmet_train/yolo_helmet_train/anno/train.txt'
    # 要检测的类别
    path_voc_names = './cfg/face.names'
    batch_size = 2
    img_size = 416
    num_workers = 2
    dataset = LoadImagesAndLabels(path, batch_size, img_size=img_size, augment=False, multi_scale=False,
                                  root_path=root_path)
    # print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                            collate_fn=dataset.collate_fn)
    # for a in enumerate(dataloader):
    #     pass
    # for i, (imgs, targets, img_path_, _) in enumerate(dataloader):
    #     print('标注信息', len(targets), targets)
    #     print(imgs.size, '===========================')
    #     for j in range(batch_size):
    #         img_tmp = np.uint8(imgs[j].permute(1, 2, 0) * 255.0)[:, :, ::-1]
    #         img_tmp = np.ascontiguousarray(img_tmp)
    #         out_path = os.path.join("/root/demo_img/", os.path.basename(img_path_[j]))
    #         for k in range(len(targets)):
    #             anno = targets[k][1::]
    #             # print(anno, 'anno')
    #             label = int(anno[0])
    #
    #             # 获取框的坐标值，左上角坐标和右下角坐标
    #             x1 = int((float(anno[1]) - float(anno[3]) / 2) * img_size)
    #             y1 = int((float(anno[2]) - float(anno[4]) / 2) * img_size)
    #
    #             x2 = int((float(anno[1]) + float(anno[3]) / 2) * img_size)
    #             y2 = int((float(anno[2]) + float(anno[4]) / 2) * img_size)
    #
    #             # 将标注框绘制在图像上
    #             cv2.rectangle(img_tmp, (x1, y1), (x2, y2), (255, 30, 30), 2)
    #             # 将标注类别绘制在图像上
    #             # cv2.putText(img_tmp, ("%s" % (str('face'))), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 55), 6)
    #             cv2.putText(img_tmp, ("%s" % (str("face"))), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 55, 255), 2)
    #         cv2.imwrite(out_path, img_tmp)
    #         print(out_path)
    #     # continue
    #     # print('结束')
    #     break
    # 第一步：指定文件路径
    path = '/www/dataset/yolo_helmet_train/yolo_helmet_train/anno/train.txt'
    path_voc_names = '../cfg/face.names'
    # 第二步：获取目标类别
    with open(path_voc_names, 'r', encoding='utf*8') as f:
        lable_map = f.readlines()
    for i in range(len(lable_map)):
        lable_map[i] = lable_map[i].strip()
        print(i, lable_map[i])
    # 第三步：获取图像数据和标注信息
    with open(path, 'r', encoding='utf-8') as file:
        img_files = file.readlines()
    for i in range(len(img_files)):
        img_files[i] = img_files[i].strip()
        print(img_files[i])

    label_files = [x.replace('images', 'labels').replace('.jpg', '.txt') for x in img_files]
    print(label_files)
    # 第四步：将标注信息绘制在图像上
    for i in range(len(label_files)):
        # 获取图像文件
        img_file = os.path.join('/www/dataset/yolo_helmet_train/', img_files[i][2:])
        out_path = os.path.join("/root/demo_img/", os.path.basename(img_file))

        img = cv2.imread(img_file)
        w = img.shape[1]
        h = img.shape[0]
        # 标注文件
        lable_path = os.path.join('/www/dataset/yolo_helmet_train/', label_files[i][2:])
        # 读取标注绘制在图像上
        if os.path.isfile(lable_path):
            with open(lable_path, 'r') as file:
                lines = file.read().splitlines()
            x = np.array([x.split() for x in lines], dtype=np.float32)
            for k in range(len(x)):
                anno = x[k]
                label = int(anno[0])
                x1 = int((float(anno[1]) - float(anno[3]) / 2) * w)
                y1 = int((float(anno[2]) - float(anno[4]) / 2) * h)

                x2 = int((float(anno[1]) + float(anno[3]) / 2) * w)
                y2 = int((float(anno[2]) + float(anno[4]) / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 20, 20), 2)
                cv2.putText(img, ("%s" % (str(lable_map[label]))), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 0),
                            6)
        cv2.imwrite(out_path, img)
    cv2.destroyAllWindows()
