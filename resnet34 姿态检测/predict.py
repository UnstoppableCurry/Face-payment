import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import math
import cv2
import torch.nn.functional as F

from util.common_utils import *
from model import resnet50, resnet34, resnet18
import matplotlib.pyplot as plt


def model_predict(ops, model_, img):
    with torch.no_grad():
        idx = 0
        # 3.数据加载
        # 遍历文件夹
        # 图像宽高
        img_width = img.shape[1]
        img_height = img.shape[0]
        # resize
        # 输入图片尺寸调整
        img_ = cv2.resize(img, (ops.img_size[1], ops.img_size[0]), interpolation=cv2.INTER_CUBIC)
        # 类型转换
        img_ = img_.astype(np.float32)
        # 归一化
        img_ = (img_ - 128.) / 256.
        # HWC->CHW
        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_)
        # 增加一个batch通道(bs, 3, h, w)
        img_ = img_.unsqueeze_(0)
        # 4.模型预测
        if use_cuda:
            img_ = img_.cuda()
        pre_ = model_(img_)
        out_put = pre_.cpu().detach().numpy()
        out_put = np.squeeze(out_put)
        yaw, pitch, roll = out_put
        yaw = yaw * 90
        pitch = pitch * 90
        roll = roll * 90
        print("yaw: {:.1f}, pitch: {:.1f}, roll: {:.1f}".format(yaw, pitch, roll))
        return [yaw, pitch, roll]


if __name__ == "__main__":
    # 1.配置信息解析
    parser = argparse.ArgumentParser(description=' Project face euler angle Test')
    # 训练好的模型路径
    parser.add_argument('--test_model', type=str,                                  #yaw: -17.7, pitch: 14.9, roll: 7.3
                        # default='./model_exp/resnet_18_imgsize_256-epoch-15.pth',#yaw: -4.3, pitch: 10.6, roll: -3.1
                        default='./model_exp/2022-01-24_00-46-50/resnet_18_imgsize_256-epoch-5.pth',
                        help='test_model')
    # 模型类型
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='model : resnet_x')
    # 分类类别个数
    parser.add_argument('--num_classes', type=int, default=3,
                        help='num_classes')
    # GPU选择
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')
    # 测试集路径
    parser.add_argument('--test_path', type=str, default='/root/cv/dataset/人脸/datasets/face_euler_angle_datasets_mini/',
                        help='test_path')
    # 输入模型图片尺寸
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')
    # 是否可视化图片
    parser.add_argument('--vis', type=bool, default=True,
                        help='vis')
    print('\n/******************* {} ******************/\n'.format(parser.description))
    # --------------------------------------------------------------------------
    # 解析添加参数
    ops = parser.parse_args()
    # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    unparsed = vars(ops)
    # 打印参数配置信息
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))
    # 设备设置
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    # 测试图片文件夹路径
    test_path = ops.test_path
    # 2.模型加载
    # 第一步：构建模型
    if ops.model == "resnet-50":
        model_ = resnet50(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == "resnet-34":
        model_ = resnet34(num_classes=ops.num_classes, img_size=ops.img_size[0])
    else:
        model_ = resnet18(num_classes=ops.num_classes, img_size=ops.img_size[0])
    # 第二步：获取设备信息
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model_.to(device)
    model_.eval()
    # 第三步：加载预训练模型
    if os.access(ops.test_model, os.F_OK):
        ckpt = torch.load(ops.test_model, map_location=device)
        model_.load_state_dict(ckpt)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    # 获取属性
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    while (cap.isOpened()):
        ret, frame = cap.read()  # 获取每一帧图像
        if ret == True:
            result = model_predict(ops, model_, frame)
            img = cv2.putText(frame, "yaw:{:.1f},pitch:{:.1f},roll:{:.1f}".format(result[0], result[1], result[2]), (1, 80),
                              cv2.FONT_HERSHEY_DUPLEX, 1,
                              (55, 0, 220), 5)
            # img = cv2.putText(frame, "ypr:{:.1f},{:.1f},{:.1f}".format(result[0], result[1], result[2]), (1, 80),
            #                   cv2.FONT_HERSHEY_DUPLEX, 2,
            #                   (255, 50, 50), 2)
            if result is None:
                continue
            cv2.imshow('result', img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
