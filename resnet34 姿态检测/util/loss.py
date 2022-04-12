# -*-coding:utf-8-*-
# date:2019-05-20
# function: wing loss
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math


def wing_loss(landmarks, labels, w=10., epsilon=2.):
    """
    :param landmarks: 预测值
    :param labels: 真实值
    :param w:
    :param epsilon:
    :return:
    """
    # MAE损失
    x = landmarks - labels
    # 插值
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    # 绝对值
    absolute_x = torch.abs(x)
    # 计算损失值
    losses = torch.where((w > absolute_x), w * torch.log(1.0 + absolute_x / epsilon), absolute_x - c)
    # 损失平均
    losses = torch.mean(losses, dim=1, keepdim=True)
    loss = torch.mean(losses)
    # 返回损失值
    return loss


def got_total_wing_loss(output, crop_landmarks):
    """
    获取总损失
    :param output:
    :param crop_landmarks:
    :return:
    """
    loss = wing_loss(output, crop_landmarks)

    return loss
