import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from util.datasets import *
from util.loss import *
from model import resnet50, resnet34, resnet18
import cv2
import time
import json
from datetime import datetime
import random
# 导入制图工具包
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from util.common_utils import mkdir_

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def trainer(ops):
    # 设备信息
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    #  构建模型
    # 2.模型加载
    # 第一步：模型结构初始化
    if ops.model == 'resnet_50':
        model_ = resnet50(pretrained=False, num_classes=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    elif ops.model == 'resnet_34':
        model_ = resnet34(pretrained=False, num_classes=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    elif ops.model == 'resnet_18':
        model_ = resnet18(pretrained=False, num_classes=ops.num_classes, img_size=ops.img_size[0],
                          dropout_factor=ops.dropout)
    # GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    # 第二步：加载预训练模型
    if os.access(ops.fintune_model, os.F_OK):  # checkpoint
        chkpt = torch.load(ops.fintune_model, map_location=device)
        model_.load_state_dict(chkpt)

    # 3.数据加载
    dataset = LoadImagesAndLabels(ops=ops, img_size=ops.img_size, flag_agu=ops.flag_agu)
    dataloader = DataLoader(dataset, batch_size=ops.batch_size, num_workers=ops.num_workers, shuffle=True)

    # 4.模型训练
    # 第一步：相关参数设置
    # 优化器设置
    optimizer = torch.optim.Adam(model_.parameters(), lr=ops.init_lr, betas=(0.9, 0.99), weight_decay=ops.weight_decay)

    # 损失函数设置
    if ops.loss_define != "wing_loss":
        criterion = nn.MSELoss(reduce=True, reduction='mean')
    # 相关参数初始化
    loss_list = []
    # 第二步：遍历每个epoch开始进行训练
    for epoch in range(0, ops.epochs):
        model_.train()
        loss_mean = 0
        loss_idx = 0
        # 第三步：遍历batch中的数据，进行预测
        for i, (imgs_, angles_) in enumerate(dataloader):
            if use_cuda:
                imgs_ = imgs_.cuda()
                angles_ = angles_.cuda()
            output_angles = model_(imgs_.float())

            # 损失计算
            if ops.loss_define == "wing_loss":
                loss_angles = got_total_wing_loss(output_angles, angles_.float())
            else:
                loss_angles = criterion(output_angles, angles_.float())
            loss = loss_angles
            loss_list.append(loss)
            loss_mean += loss.item()
            loss_idx += 1
            # 打印结果
            # 每10个batch打印一次结果
            if i % 1 == 0:
                loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('  %s - %s - epoch [%s/%s] (%s/%s):' % (
                    loc_time, ops.model, epoch, ops.epochs, i, int(dataset.__len__() / ops.batch_size)), \
                      'Mean Loss : %.6f - Loss: %.6f' % (loss_mean / loss_idx, loss.item()), ' bs:', ops.batch_size, \
                      ' img_size: %s x %s' % (ops.img_size[0], ops.img_size[1]))

            # 反向传播进行梯度更新
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 第四步：保存ckpt
        if epoch % 5 == 0:
            torch.save(model_.state_dict(),
                       ops.model_exp + '{}_imgsize_{}-epoch-{}.pth'.format(ops.model, ops.img_size[0], epoch))

    # 第五步：损失变化的曲线
    plt.figure(0)
    plt.plot(loss_list, color="blue", lable="loss")
    plt.savefig('./loss.png')


# 1、配置信息设置
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' Project Face Euler Angle Train')
    # 模型输出文件夹
    parser.add_argument('--model_exp', type=str, default='./model_exp',
                        help='model_exp')
    # 模型类型
    parser.add_argument('--model', type=str, default='resnet_18',
                        help='model : resnet_18,resnet_34,resnet_50')
    #  yaw,pitch,roll
    parser.add_argument('--num_classes', type=int, default=3,
                        help='num_classes')
    # GPU选择
    parser.add_argument('--GPUS', type=str, default='0',
                        help='GPUS')
    # 训练集标注信息
    parser.add_argument('--train_path', type=str,
                        default='/root/cv/dataset/人脸/datasets/face_euler_angle_datasets_mini/',
                        help='train_path')
    # 是否使用预训练模型
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='imageNet_Pretrain')
    # 预训练模型位置
    parser.add_argument('--fintune_model', type=str,
                        default='/www/model/resnet-预训练模型/resnet18/resnet18-5c106cde.pth',
                        help='fintune_model')
    # 损失函数定义
    parser.add_argument('--loss_define', type=str, default='wing_loss',
                        help='define_loss')
    # 初始化学习率
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='init_learningRate')
    # 优化器正则损失权重
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight_decay')
    # 优化器动量
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # 训练每批次图像数量
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    # dropout
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout')
    # 训练周期
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')
    # 训练线程数
    parser.add_argument('--num_workers', type=int, default=200,
                        help='num_workers')
    # 输入模型图片尺寸
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='img_size')
    # 是否进行数据增强
    parser.add_argument('--flag_agu', type=bool, default=True,
                        help='data_augmentation')
    # 模型输出文件夹是否进行清除
    parser.add_argument('--clear_model_exp', type=bool, default=False,
                        help='clear_model_exp')

    # --------------------------------------------------------------------------
    args = parser.parse_args()  # 解析添加参数
    # --------------------------------------------------------------------------
    # 根据配置信息创建训练结果保存的根目录
    # mkdir_的功能是：
    # 存在路径时
    # 若flag_rm = True，则删除文件重新创建
    # 否则不修改
    # 若不存在路劲，则创建路径即可
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time) + '/'
    # 根据训练时间创建保存结果的路经
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    unparsed = vars(args)
    # 打印参数结果
    for key in unparsed.keys():
        print('{} : {}'.format(key, unparsed[key]))
    # 当前时间
    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)
    # 将配置信息写入到文件中
    fs = open(args.model_exp + 'train_ops.json', "w", encoding='utf-8')
    json.dump(unparsed, fs, ensure_ascii=False, indent=1)
    fs.close()
    # 模型训练
    trainer(ops=args)
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
