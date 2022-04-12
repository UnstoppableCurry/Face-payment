import os
from yoloV3 import Yolov3, Yolov3Tiny
from util.parse_config import parse_data_cfg
from util.torch_utils import select_device
import torch
from torch.utils.data import DataLoader
from util.datasets import LoadImagesAndLabels
from util.utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from util.utils import *


def train(data_config='cfg/face.data'):
    # 1.配置文件解析
    get_data_cfg = parse_data_cfg(data_config)
    gpus = get_data_cfg['gpus']
    num_workers = int(get_data_cfg['num_workers'])
    cfg_model = get_data_cfg['cfg_model']
    train_path = get_data_cfg['train']
    num_classes = int(get_data_cfg['classes'])
    finetune_model = int(get_data_cfg['batch_size'])
    batch_size = int(get_data_cfg['batch_size'])
    img_size = int(get_data_cfg['img_size'])
    multi_scale = get_data_cfg['multi_scale']
    epochs = int(get_data_cfg['epochs'])
    lr_step = str(get_data_cfg['lr_step'])
    lr0 = float(get_data_cfg['lr0'])
    # root_path = str(get_data_cfg['root_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = select_device()
    if multi_scale == 'True':
        multi_scale = True
    else:
        multi_scale = False

    # 2 模型加载
    if '-tiny' in cfg_model:
        model = Yolov3Tiny(num_classes)
        print('nimi')
        weights = './weights-yolov3-face-tiny4/'
    else:
        model = Yolov3(num_classes)
        weights = './weights-yolov3-face3-heat'
    model = model.to(device)
    # 设置模型训练位置
    if not os.path.exists(weights):
        os.mkdir(weights)
    latest = weights + 'latest_{}.pt'.format(img_size)
    # 设置优化器和学习率衰减策略
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)
    milestones = [int(i) for i in lr_step.split(',')]
    print(milestones, '动态学习率变换')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=-1)

    # 数据加载

    dataset = LoadImagesAndLabels(train_path, batch_size=batch_size, img_size=img_size, augment=True,
                                  multi_scale=multi_scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            collate_fn=dataset.collate_fn, drop_last=True)
    flag_start = False
    xy_loss = []
    wh_loss = []
    conf_loss = []
    cls_loss = []
    total_loss = []
    # 遍历数据集开始训练
    for epoch in range(0, epochs):
        model.train()
        if flag_start:  # 第一次循环 学习率不变化
            scheduler.step()
        flag_start = True
        mloss = defaultdict(float)  # loss初始化，定义一个字典
        # 模型训练
        t = time.time()
        nb = len(dataloader)
        print('nb-->', nb)
        n_burnin = min(round(nb / 5 + 1), 1000)
        print(n_burnin)
        # 遍历每一个batchsize的数据
        for i, (imgs, taegets, img_path, _) in enumerate(dataloader):
            multi_size = imgs.size()
            imgs = imgs.to(device)
            taegets = taegets.to(device)
            nt = len(taegets)
            if nt == 0:
                # 如果没有标签就跳过
                continue
            # 学习率预热
            # if epoch > 0:
            if epoch == 0 and i <= n_burnin:
                lr = lr0 * (i / n_burnin) ** 4  # 当起始step比较小时,并且是第一个epoch时,用比较小的学习率
                # lr = lr0 * epoch * i*8 / n_burnin  # 当起始step比较小时,并且是第一个epoch时,用比较小的学习率
                print(lr)
                for x in optimizer.param_groups:
                    x['lr'] = lr  # 将lr更新到优化器中
            # 模型预测
            pred = model(imgs)
            target_list = build_targets(model, taegets)
            # loss
            loss, loss_dict = compute_loss(pred, target_list,nt,batch_size)
            # xywh mse 置信度用BCEWithLogitsLoss多分类损失  cls用二分类交叉熵损失CrossEntropyLoss
            # 老三样
            loss.backward()  # 反向传播 计算题都
            optimizer.step()  # 梯度更新
            optimizer.zero_grad()  # 梯度清零
            # 打印 平均损失
            for key, value in loss_dict.items():
                mloss[key] = (mloss[key] * i + value) / (i + 1)
            print(
                '  Epoch {:3d}/{:3d}, Batch {:6d}/{:6d}, Img_size {}x{}, nTargets {}, lr {:.6f}, loss: xy {:.3f}, wh {:.3f}, '
                'conf {:.3f}, cls {:.3f}, total {:.3f}, time {:.3f}s'
                    .format(epoch, epochs - 1, i, nb - 1,
                            multi_size[2], multi_size[3]
                            , nt, scheduler.get_lr()[0],
                            mloss['xy'], mloss['wh'],
                            mloss['conf'], mloss['cls'],
                            mloss['total'], time.time() - t), end='\n')
            xy_loss.append(mloss['xy'])
            wh_loss.append(mloss['wh'])
            conf_loss.append(mloss['conf'])
            cls_loss.append(mloss['cls'])
            total_loss.append(mloss['total'])
            t = time.time()
        # 模型保存
        chkpt = {
            'epoch': epoch,
            'model': model.module.state_dict() if type(
                model) is nn.parallel.DistributedDataParallel else model.state_dict()
        }
        torch.save(chkpt, weights + '/yolov3_last_{}_epoch_{}.pt'.format(img_size, epoch))
        # 创建第一张画布
        plt.figure(0)

        # 绘制坐标损失曲线
        plt.plot(xy_loss, label="xy Loss")
        # 绘制宽高损失曲线 , 颜色为红色
        plt.plot(wh_loss, color="red", label="wh Loss")
        # 绘制置信度损失曲线 , 颜色为绿色
        plt.plot(conf_loss, color="green", label="conf Loss")
        # 绘制分类损失曲线 , 颜色为绿色
        plt.plot(cls_loss, color="yellow", label="cls Loss")
        # 绘制总损失曲线 , 颜色为蓝色
        plt.plot(total_loss, color="blue", label="sum Loss")
        # 曲线说明在左上方
        # plt.legend(loc='upper left')
        # 保存图片
        plt.savefig(weights + '/yolov3_last_{}_epoch_{}_loss.png'.format(img_size, epoch))
        del chkpt

    # 创建第一张画布
    plt.figure(0)

    # 绘制坐标损失曲线
    plt.plot(xy_loss, label="xy Loss")
    # 绘制宽高损失曲线 , 颜色为红色
    plt.plot(wh_loss, color="red", label="wh Loss")
    # 绘制置信度损失曲线 , 颜色为绿色
    plt.plot(conf_loss, color="green", label="conf Loss")
    # 绘制分类损失曲线 , 颜色为绿色
    plt.plot(cls_loss, color="yellow", label="cls Loss")
    # 绘制总损失曲线 , 颜色为蓝色
    plt.plot(total_loss, color="blue", label="sum Loss")
    # 曲线说明在左上方
    plt.legend(loc='upper left')
    # 保存图片
    plt.savefig(weights+"./loss.png")


if __name__ == '__main__':
    train(data_config='cfg/face.data')
    print('完成')
#


# train=/www/dataset/yolo_helmet_train/yolo_helmet_train/anno/train.txt
# valid=/www/dataset/yolo_helmet_train/yolo_helmet_train/anno/train.txt
