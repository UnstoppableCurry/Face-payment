# -*-coding:utf-8-*-
# date:2020-04-25
# Author: Eric.Lee
# function: inference

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F

from model import resnet50, resnet34, resnet34_2
from util.common_utils import *
import copy
import matplotlib.pyplot as plt


# 反归一化
def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


if __name__ == "__main__":
    # 1.配置信息解析
    parser = argparse.ArgumentParser(description=' Project Landmarks Test')
    # 模型路径
    parser.add_argument('--test_model', type=str,
                        # default='/root/cv/pycharm/人脸检测/人脸多任务/model_exp/2022-01-28_09-14-08/resnet_34_epoch-8.pth',
                        # default='/root/cv/pycharm/人脸检测/人脸多任务/model_exp/2022-01-28_08-39-03/resnet_34_epoch-9.pth',
                        # default='/root/cv/dataset/人脸/facetoPay/face_multi_task/face_multitask-resnet_34_imgsize-256-20210425.pth',
                        # default='/root/cv/pycharm/人脸检测/人脸多任务/model_exp/2022-01-28_22-04-11/resnet_34_epoch-80.pth',
                        # 正常
                        # default='/root/cv/pycharm/人脸检测/人脸多任务/model_exp/2022-01-28_22-42-34/resnet_34_epoch-22.pth',
                        default='/root/cv/pycharm/人脸检测/人脸多任务/model_exp/2022-01-29_01-38-27/resnet_34_epoch-98.pth',#开图数据增强后,无训练模型加载
                        # default='/root/cv/pycharm/人脸检测/人脸多任务/model_exp/2022-01-28_11-23-49/resnet_34_epoch-1.pth',

                        help='test_model')
    # 模型类型
    parser.add_argument('--model', type=str, default='resnet_34',
                        help='model : resnet_50')
    # 输出数据（关键点）的个数
    parser.add_argument('--num_classes', type=int, default=196,
                        help='num_classes')
    # GPU选择
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')
    # 测试集路径
    parser.add_argument('--test_path', type=str,
                        default='/root/cv/dataset/人脸/facetoPay/face_multi_task/img/',
                        help='test_path')
    # 输入模型图片尺寸
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')
    # 是否可视化图片
    parser.add_argument('--vis', type=bool, default=True,
                        help='vis')
    ops = parser.parse_args()  # 解析添加参数
    # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    unparsed = vars(ops)
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))
    # 设备信息
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    # 测试图片文件夹路径
    test_path = ops.test_path

    # 2.模型加载
    # 第一步：构建模型
    # 加载模型
    if ops.model == 'resnet_50':
        model_ = resnet50(landmarks_num=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(landmarks_num=ops.num_classes, img_size=ops.img_size[0])
    elif ops.mode == 'resnet_34_2':
        model_ = resnet34_2(landmarks_num=ops.num_classes, img_size=ops.img_size[0])

    # 第二步：获取设备信息，并将模型写入设备中
    # 设备设置
    use_cuda = False  # torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    # 设置为前向推断模式
    model_.eval()

    # 第三步：加载预训练模型
    # 加载训练好的模型
    if os.access(ops.test_model, os.F_OK):
        chkpt = torch.load(ops.test_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.test_model))
    # 预测图片
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 不进行梯度更新
    with torch.no_grad():
        # 3.数据加载
        # 记录处理图片的个数
        idx = 0
        # 遍历文件夹
        for file in os.listdir(ops.test_path):
            # 若不以jpg结尾，则进行下次循环
            if '.jpg' not in file:
                continue
            idx += 1
            print('{}) image : {}'.format(idx, file))
            # 读取图像数据
            img = cv2.imread(ops.test_path + file)
            # 获取图像的宽高
            img_width = img.shape[1]
            img_height = img.shape[0]
            # 输入图片预处理，修正图像的大小
            img_ = cv2.resize(img, (ops.img_size[1], ops.img_size[0]))
            # 类型转换和归一化
            img_ = img_.astype(np.float32)
            img_ = (img_ - 128.) / 256.
            # 通道调整
            img_ = img_.transpose(2, 0, 1)
            img_ = torch.from_numpy(img_)
            # 增加batch维
            img_ = img_.unsqueeze_(0)
            # 4.模型预测
            # 将数据写入设备中
            if use_cuda:
                img_ = img_.cuda()
            # 模型预测
            if ops.model == 'resnet_34':
                output_landmarks, output_gender, output_age, output_face_ol = model_(img_.float())
            elif ops.model == 'resnet_34_2':
                output_landmarks, output_gender, output_age = model_(img_.float())

            # output_landmarks, output_gender, output_age, output_face_ol = model_(img_.float())
            # 获取关键点预测结果
            output_landmarks = output_landmarks.cpu().detach().numpy()
            # 去除batch维
            output_landmarks = np.squeeze(output_landmarks)
            # 获取关键点，以字典的形式输出，每个关键点不绘制圆形
            dict_landmarks = draw_landmarks(img, output_landmarks, draw_circle=False)
            # 绘制关键点
            draw_contour(img, dict_landmarks)
            # 性别输出结果
            output_gender = F.softmax(output_gender, dim=1)
            output_gender = output_gender[0]
            output_gender = output_gender.cpu().detach().numpy()
            output_gender = np.array(output_gender)
            # 概率最大类别索引，获取对应的性别
            gender_max_index = np.argmax(output_gender)
            # 最大概率
            score_gender = output_gender[gender_max_index]
            print(gender_max_index, score_gender)

            # 年龄输出结果
            output_age = output_age.cpu().detach().numpy()[0][0]
            output_age = (output_age * 100. + 50.)
            img = cv2.resize(img, (ops.img_size[1], ops.img_size[0]))

            # 将预测结果绘制图像上
            if gender_max_index == 1.:
                cv2.putText(img, 'gender:{}'.format("male"), (2, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, 'gender:{}'.format("male"), (2, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 20, 0), 1)
            else:
                cv2.putText(img, 'gender:{}'.format("female"), (2, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(img, 'gender:{}'.format("female"), (2, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 20, 0), 1)
            cv2.putText(img, 'age:{:.2f}'.format(output_age), (2, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, 'age:{:.2f}'.format(output_age), (2, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 20, 0), 1)
            # cv2.putText(img, "yaw:{:.1f},pitch:{:.1f},roll:{:.1f}".format(output_face_ol[0][0], output_face_ol[0][1],
            #                                                               output_face_ol[0][2]), (1, 80),
            #             cv2.FONT_HERSHEY_COMPLEX, 0.8,
            #             (55, 0, 220), 1)
            # 将预测结果展示出来或保存下来
            if ops.vis:
                # cv2.namedWindow('image', 0)
                # cv2.imshow('image', img)
                # cv2.waitKey(0)
                # if cv2.waitKey(1) == 27:
                #     break
                # cv2.imwrite("./result2/" + file, img)
                plt.imshow(img[:, :, ::-1])
                plt.show()
            cv2.destroyAllWindows()

            print('well done ')
