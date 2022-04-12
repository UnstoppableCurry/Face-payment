from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans


# 模型训练时的参数配置信息
def get_config():
    conf = edict()
    conf.name = ''
    # 训练的轮次
    conf.epochs = 50
    # 工作目录：
    conf.work_path = "./"
    # 微调模型的存储位置
    # conf.finetune_backbone_model = "/root/cv/pycharm/人脸检测/人脸识别/save/model_2022-01-31-02-22-13_step_19188.pth"
    # conf.finetune_head_model = "/root/cv/pycharm/人脸检测/人脸识别/save/head_2022-01-31-02-22-13_step_19188.pth"
    conf.finetune_backbone_model = ""
    conf.finetune_head_model = ""

    # 训练集路径
    # 训练集路径
    # conf.datasets_train_path = "C:\\Users\86183\Desktop\人脸识别数据集\\"
    # conf.datasets_train_path = "/root/cv/dataset/人脸/datasets/insight_face"
    conf.datasets_train_path = "/root/cv/pycharm/人脸检测/人脸识别/mydataset2"
    # 模型结果保存位置
    conf.save_path = conf.work_path + 'local_save'
    # 图像大小
    conf.input_size = [112, 112]
    # 特征向量的大小
    conf.embedding_size = 512
    # 网络深度
    conf.net_depth = 50
    # 随机失活的概率
    conf.drop_ratio = 0.6
    # 模型模式
    conf.net_mode = 'ir_se'  # or 'ir'
    # 设备信息
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # bacth大小
    conf.batch_size = 128
    # conf.batch_size = 128
    # 学习率
    conf.lr = 1e-3
    # 步进式衰减的轮次
    conf.milestones = [12, 15, 18]
    # 动量
    conf.momentum = 0.9
    conf.num_workers = 6
    # 损失函数
    conf.ce_loss = CrossEntropyLoss()
    return conf
