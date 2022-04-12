from datetime import datetime
from PIL import Image
import numpy as np
import io
from torchvision import transforms as trans
import torch
import sys, os

# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(root_path)
# print(root_path)
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from model import l2_norm
from util.datasets import de_preprocess

import pdb
import cv2


# 获取网络中的不同层
def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


# 将反归一化，类型转换，翻转，归一化处理合并在一起进行
hflip = trans.Compose([
    de_preprocess,
    trans.ToPILImage(),
    trans.functional.hflip,
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


# 绘制框及分类结果
def draw_box_name(bbox, name, frame):
    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
    frame = cv2.putText(frame,
                        name,
                        (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)
    return frame


# 学习率衰减策略
def schedule_lr(optimizer):
    # 每次变为原来的0.1
    for params in optimizer.param_groups:
        params['lr'] /= 10
    print(optimizer)
