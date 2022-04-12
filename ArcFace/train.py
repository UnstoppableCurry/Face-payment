import os
import warnings

warnings.filterwarnings("ignore")

from config import get_config
import argparse
# --------------------------------
from util.datasets import de_preprocess, get_train_loader
from model import Backbone, Arcface, l2_norm
import torch
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from util.utils import separate_bn_paras, schedule_lr
from PIL import Image
from torchvision import transforms as trans
import math
import time
from torch.utils.tensorboard import SummaryWriter


def trainer(conf):
    tb_writer = SummaryWriter(comment=conf.name)
    # 加载训练集数据
    data_loader, class_num, datasets_len = get_train_loader(conf)
    # 模型选择
    model_ = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
    print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
    if os.access(conf.finetune_backbone_model, os.F_OK):
        model_.load_state_dict(torch.load(conf.finetune_backbone_model))
        print("-------->>>   load model : {}".format(conf.finetune_backbone_model))
    # 加载head模型
    head_ = Arcface(embedding_size=conf.embedding_size, classnum=class_num).to(conf.device)

    if os.access(conf.finetune_head_model, os.F_OK):
        head_.load_state_dict(torch.load(conf.finetune_head_model))
        print("-------->>>   load head : {}".format(conf.finetune_head_model))
    # 优化器
    paras_only_bn, paras_wo_bn = separate_bn_paras(model_)
    optimizer = optim.SGD([
        {'params': paras_wo_bn + [head_.kernel], 'weight_decay': 5e-4},
        {'params': paras_only_bn}
    ], lr=conf.lr, momentum=conf.momentum)
    # bn层分离冻结不进行正则化
    model_.train()
    step_ = 0
    # 用来存放loss，和准确率进行绘图
    loss_list = []
    tensorboard_step = 0
    loss_mean = 0
    # 遍历每一个epoch
    for e in range(conf.epochs):
        # 学习率衰减策略，变为原来的0.1倍
        print("  epoch < {} >".format(e))
        if e == conf.milestones[0]:
            schedule_lr(optimizer)
        if e == conf.milestones[1]:
            schedule_lr(optimizer)
        if e == conf.milestones[2]:
            schedule_lr(optimizer)
        for i, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            optimizer.zero_grad()
            embeddings = model_(imgs)
            thetas = head_(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)
            loss_list.append(loss)
            loss.backward()
            optimizer.step()
            # 每10个迭代次数打印信息
            if i % 10 == 0:
                print(
                    "  epoch - < {}/{} >, [{}/{}], loss: {:.6f} , bs: {}".format(
                        e, conf.epochs, i, int(datasets_len / conf.batch_size), loss.item(),
                        conf.batch_size))
            # 每100个迭代次数保存checkpoint
            # if step_ % 533 == 0:
            # 迭代次数加1
            step_ += 1
            tensorboard_step += 1
            loss_mean += loss.item()
            if tb_writer:
                tb_writer.add_scalar("epoch", e, tensorboard_step)
                tb_writer.add_scalar("tensorboard_step", tensorboard_step, tensorboard_step)
                tb_writer.add_scalar("loss", loss.item(), tensorboard_step)
                tb_writer.add_scalar("mean_loss", loss_mean / tensorboard_step, tensorboard_step)
                # 保存路径
        save_path = conf.save_path
        # 若不存在，则创建该路径
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            # 获取当前时刻
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # 保存backbone的结果
        torch.save(
            model_.state_dict(), save_path +
                                 ('/model_{}_step_{}.pth'.format(time_str, step_)))
        # 保存head部分的结果
        torch.save(
            head_.state_dict(), save_path +
                                ('/head_{}_step_{}.pth'.format(time_str, step_)))
        # 创建第一张画布
        plt.figure(0)
        # 绘制总损失曲线 , 颜色为蓝色
        plt.plot(loss_list, color="blue", label="Loss")
        # 曲线说明在左上方
        plt.legend(loc='upper left')
        # 保存图片
        plt.savefig("./loss.png")


if __name__ == '__main__':
    # 获取配置信息
    conf = get_config()
    # 模型训练
    trainer(conf)
