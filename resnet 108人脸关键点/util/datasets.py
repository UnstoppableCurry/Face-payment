import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
import os
import random
import math
import glob
from torch.utils.data import Dataset
import sys
# from data_agu import *
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from util.data_agu import *

# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(root_path)
# from data_agu import *
# print(root_path)

class LoadImagesAndLabels(Dataset):
    def __init__(self, ops, img_size=(224, 224), flag_agu=False):
        '''
        数据初始化
        :param ops:配置信息对象
        :param img_size: 图像尺寸
        :param flag_agu: 标记位
        '''
        max_age = 0  # 年龄最大值在0以上
        min_age = 65535  # 年龄最小值在65535以下
        file_list = []  # 图像文件list
        landmarks_list = []  # 关键点list
        age_list = []  # 年龄list
        gender_list = []  # 性别list
        idx = 0  # 下标图像计数
        for f_ in os.listdir(ops.train_path):
            # 读取json解析
            f = open(ops.train_path + f_, encoding='utf8')
            dict = json.load(f)
            f.close()
            if dict['age'] > 100. or dict['age'] < 1.:
                continue  # 年龄数据错误
            idx += 1
            img_path_ = (ops.train_path + f_).replace('label_new', 'image').replace('.json', '.jpg')
            # img = cv2.imread(img_path_)
            file_list.append(img_path_)  # 这为啥还存
            pts = []  # 存储关键点
            for pt_ in dict['landmarks']:
                x, y = pt_
                pts.append([x, y])
            landmarks_list.append(pts)
            if dict['gender'] == 'male':  # 存储性别
                gender_list.append(1)
            else:
                gender_list.append(0)
            age_list.append(dict['age'])  # 存储年龄
            if max_age < dict['age']:
                max_age = dict['age']
            if min_age > dict['age']:
                min_age = dict['age']
        self.files = file_list
        self.landmarks = landmarks_list
        self.genders = gender_list
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.ages = age_list

    def __len__(self):
        # 获取图像的个数
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        pts = self.landmarks[index]  # 关键点
        gender = self.genders[index]  # 性别
        age = self.ages[index]  # 年龄

        # 读取图像
        img = cv2.imread(img_path)
        if self.flag_agu and random.random() > 0.35:  # 如果进行图像增强,进行图像旋转
            angle_random = random.randint(-33, 33)  # 随机生成旋转角度
            left_eye = np.average(pts[60:68], axis=0)  # 获取左眼和右眼的关键点均值,用于计算旋转中心
            right_eye = np.average(pts[68:76], axis=0)
            # 返回旋转后的crop图和归一化的关键点
            img_, landmarks_ = face_random_rotate(img, pts, angle_random, left_eye, right_eye, img_size=self.img_size)
        else:
            # 对人脸区域进行裁剪,并归一化
            # 人脸区域裁剪
            x_max = -65535
            y_max = -65535
            x_min = 65535
            y_min = 65535
            for pt_ in pts:
                # 获取关键点区域左上角坐标和右下角坐标
                x_, y_ = int(pt_[0]), int(pt_[1])
                x_min = x_ if x_min > x_ else x_min
                y_min = y_ if y_min > y_ else y_min
                x_max = x_ if x_max < x_ else x_max
                y_max = y_ if y_max < y_ else y_max
            # 获取人脸区域的宽高
            face_w = x_max - x_min
            face_h = y_max - y_min
            # 人脸区域进行扩展并进行裁剪
            # 对人脸区域进行随机的扩展
            x_min = int(x_min - random.randint(-6, int(face_w / 10)))
            y_min = int(y_min - random.randint(-6, int(face_h / 10)))
            x_max = int(x_max + random.randint(-6, int(face_w / 10)))
            y_max = int(y_max + random.randint(-6, int(face_h / 10)))
            # 确保坐标在图像范围内
            x_min = np.clip(x_min, 0, img.shape[1] - 1)
            x_max = np.clip(x_max, 0, img.shape[1] - 1)
            y_min = np.clip(y_min, 0, img.shape[0] - 1)
            y_max = np.clip(y_max, 0, img.shape[0] - 1)
            # 修正后的坐标 再次获取人脸区域的宽高
            face_w = x_max - x_min
            face_h = y_max - y_min
            face_cut = img[y_min:y_max, x_min:x_max, :]  # 裁切
            landmarks_ = []  # 关键点
            for pt_ in pts:
                x_, y_ = int(pt_[0]) - x_min, int(pt_[1]) - y_min  # 获取关键点左上角,右下角相对于裁切后的坐标
                landmarks_.append([float(x_) / float(face_w), float(y_) / float(face_h)])  # 归一化处理
            img_ = cv2.resize(face_cut, self.img_size, interpolation=random.randint(0, 4))  # 图像缩放
        # 第三步：图像增强
        # 颜色增强
        if self.flag_agu:
            # 颜色增强 70%的概率
            if random.random() > 0.7:
                # 颜色空间转换
                img_hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10, 10)
                # 对H通道进行增强
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_x)
                # 对取值进行修正
                img_hsv[:, :, 0] = np.maximum(img_hsv[:, :, 0], 0)
                img_hsv[:, :, 0] = np.minimum(img_hsv[:, :, 0], 180)
                # 将色彩空间转换为BGR
                img_ = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            # 对数据进行归一化,类型处理
        img_ = img_.astype(np.float32)[:, :, ::-1]  # BGR->RGB
        # 数值归一化
        img_ = (img_ - 128.0) / 256.
        img_ = img_.transpose(2, 0, 1)  # CHW->HWC 加快收敛
        # 关键点扁平化处理
        landmarks_ = np.array(landmarks_).ravel()  # flatten展平 [98,2]-->[1,196]
        age = np.expand_dims(np.array((age - 50.) / 100.), axis=0)  # 年龄归一化,添加维度 本身维度是一维,但是预测需要2维数据
        # [len,1]->[1,len,1]
        return img_, landmarks_, gender, age


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Project Mult task train')
    # 训练集标注信息
    parser.add_argument('--train_path', type=str,
                        default='/root/cv/dataset/人脸/datasets/wiki_crop_face_multi_task/label_new/', help='train_path')
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')  # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool, default=False,
                        help='data_augmentation')  # 训练数据生成器是否进行数据扩增
    ops = parser.parse_args()  # 解析添加参数
    dataset = LoadImagesAndLabels(ops, img_size=ops.img_size, flag_agu=ops.flag_agu)
    print(dataset.__len__())
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=True, drop_last=True)
    for (img_, pts_, gender_, age_) in dataloader:
        for j in range(2):
            img = np.uint8(img_[j].permute(1, 2, 0) * 256.0 + 128.0)[:, :, ::-1]
            img = cv2.UMat(img).get()
            # 把年龄绘制上
            cv2.putText(img, 'age:{:.2f}'.format(age_[j][0] * 100.0 + 50.0), (2, 20), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        (0, 255, 0), 2)
            if gender_[j] == 1:
                cv2.putText(img, 'gender:{}'.format("male"), (2, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'gender:{}'.format("female"), (2, 40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            # 关键点
            pts = pts_[j].reshape((-1, 2))
            for pt in pts:
                x_, y_ = int(pt[0] * 256), int(pt[1] * 256)
                cv2.circle(img, (x_, y_), 2, (0, 255, 0), -1)
            # cv2.imshow("result", img)
            # cv2.waitKey(0)
            # plt.imshow(img[:, :, ::-1])
            # plt.show()
    cv2.destroyAllWindows()
