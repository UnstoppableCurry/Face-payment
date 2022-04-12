import warnings

warnings.filterwarnings("ignore")
import os
import torch
from model import Backbone
import argparse
from pathlib import Path
import cv2
from torchvision import transforms as trans
from util.datasets import de_preprocess
import torch
from model import l2_norm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from collections import Counter
import matplotlib.pyplot as plt


# 加载pth,npy文件中存储的特征
def load_facebank(facebank_path):
    embeddings = torch.load(facebank_path + '/facebank.pth',
                            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    names = np.load(facebank_path + '/names.npy')
    return embeddings, names


def infer(model, device, faces, target_embs, threshold=1.2, tta=False, label=None, names=None):
    '''
    :param names:
    :param label: 目标值
    :param model: 进行预测的模型
    :param device: 设备信息
    :param faces: 要处理的人脸图像
    :param target_embs: 数据库中的人脸特征
    :param threshold: 阈值
    :param tta: 进行水平翻转的增强
    :return:
    '''
    rang_nums = 0
    right_nums = 0
    # 将类型转换和标准化合并在一起
    test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 特征向量
    embs = []
    # 遍历人脸图像
    for img in faces:
        # 若进行翻转
        if tta:
            # 镜像翻转
            mirror = trans.functional.hflip(img)
            # 模型预测
            emb = model(test_transform(img).to(device).unsqueeze(0))
            emb_mirror = model(test_transform(mirror).to(device).unsqueeze(0))
            # 获取最终的特征向量
            embs.append(l2_norm(emb + emb_mirror))
        else:
            with torch.no_grad():
                # 未进行翻转时，进行预测
                embs.append(model(test_transform(img).to(device).unsqueeze(0)))
    # 将特征拼接在一起
    source_embs = torch.cat(embs)
    # 计算要检测的图像特征与目标特征之间的差异
    # diff_1 = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
    # dist_1 = torch.sum(torch.pow(diff_1, 2), dim=1)

    diff = torch.mm(source_embs, target_embs.transpose(1, 0))
    dist = torch.pow(diff, 2) * 64

    # print('dist_1-->', dist_1)
    # print('dist-->', dist)

    # 获取差异最小值及对应的索引
    # minimum, min_idx = torch.min(dist.squeeze(), dim=0) #计算欧式距离 距离越小说明相似度越高
    minimum, min_idx = torch.max(dist.squeeze(), dim=1)  # 计算cos(x)时,值越大说明夹角越小
    # 若没有匹配成功，将索引设置为-1
    # min_idx[minimum > threshold] = -1
    min_idx[minimum < threshold] = 0
    # print(len(min_idx), len(minimum))
    dicts = {}
    dicts2 = {}
    for i in range(source_embs.shape[0]):
        if dist[i][list(names).index(label[i + 1]) - 1] < threshold:
            # 自己与自己比低于阈值,也就是自己与自己比预测错 dist[i][list(names).index(label[i + 1])]
            if label[i + 1] not in dicts:
                dicts[label[i + 1]] = 1
            else:
                dicts[label[i + 1]] += 1
            # print('低于阈值')  # 自己与自己比的总数是自己类别的训练集元素数量
        if names[min_idx[i] + 1] != label[i + 1]:
            rang_nums += 1
        else:
            right_nums += 1
        lists = dist[i][:list(names).index(label[i + 1]) - 1]
        FAR_index = 0
        for j in dist[i]:
            if j == dist[i][list(names).index(label[i + 1]) - 1]:
                continue
            else:
                if j > threshold:
                    if j not in dicts2:
                        dicts2[names[FAR_index + 1]] = 1
                    else:
                        dicts2[names[FAR_index + 1]] += 1
            FAR_index += 1

    # print(dicts, dicts2)
    return rang_nums, right_nums, dicts, dicts2


if __name__ == '__main__':
    # 配置相关参数
    parser = argparse.ArgumentParser(description='make facebank')
    # 模型
    parser.add_argument("--net_mode", help="which network, [ir, ir_se]", default='ir_se', type=str)
    # 模型深度
    parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    # 预训练模型
    parser.add_argument("--finetune_backbone_model", help="finetune_backbone_model",
                        # default="./save/model_2022-01-31-05-27-40_step_2144.pth",
                        # default="./save/model_2022-01-31-05-53-58_step_500.pth",
                        # default="/root/cv/pycharm/人脸检测/人脸识别/save/model_2022-01-31-06-22-37_step_3752.pth",
                        default="/root/cv/pycharm/人脸检测/人脸识别/save/model_2022-01-31-12-05-37_step_12864.pth",
                        # 自己训练标准训练集99epoch
                        # default="/root/cv/pycharm/人脸检测/人脸识别/local_save/model_2022-02-02-05-23-17_step_169.pth",
                        # default="face_verify-model_ir_se-50.pth",  # 老师训练模型
                        type=str)
    # 人脸仓库
    parser.add_argument("--facebank_path", help="facebank_path",
                        default="./facebank", type=str)
    # 是否进行水平翻转
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", default=False, type=bool)
    # 要进行识别的人脸
    # parser.add_argument("-example", help="example",
    # default="G://机器视觉//cv项目代码//人脸检测项目代码//facetoPay//facetoPay//insight_face//example//",
    # type=str)
    parser.add_argument("-example", help="example", default="/root/cv/pycharm/人脸检测/人脸识别/example/example2/", type=str)
    # parser.add_argument("-example", help="example", default="C:\\Users\86183\Desktop\example\\", type=str)
    # 参数解析
    args = parser.parse_args()
    # 设备信息
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型选择
    model_ = Backbone(args.net_depth, 1., args.net_mode).to(device_)
    print('{}_{} model generated'.format(args.net_mode, args.net_depth))
    # 加载预训练模型
    if os.access(args.finetune_backbone_model, os.F_OK):
        model_.load_state_dict(torch.load(args.finetune_backbone_model,
                                          map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        print("-------->>>   load model : {}".format(args.finetune_backbone_model))
    # 模型前向传播
    model_.eval()
    # 加载人脸仓库中的人脸特征及对应的名称
    targets, names = load_facebank(args.facebank_path)
    # 打印结果
    print("names : {}".format(names))
    print("targets size : {}".format(targets.size()))
    # 要识别的人脸
    faces_identify = []
    label = ['null', ]
    # 遍历要处理的图像
    img_nums = 0
    for file in os.listdir(args.example):
        # 若非图片文件，进行下一次循环
        # if not file.endswith('png'):
        #     continue
        label.append(file.split('-')[0])
        # label.append(file)
        # 读取图像数据
        img = cv2.imread(args.example + file)
        if img is None:
            continue
        # 获取图像的宽高
        x, y = img.shape[0:2]
        # 送入网络中的图像必须是112*112
        if x != 112 or y != 112:
            img = cv2.resize(img, (112, 112))
        # 将数据放入list中
        faces_identify.append(Image.fromarray(img))
        img_nums += 1
        # 进行检测，results是索引，face_dst是差异
    阈值 = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 24, 29, 34, 39, 45, 49]
    yuzhi = np.arange(0.1, 100, 0.1)
    # yuzhi = [x for x in range(1, 50)]
    frr = []
    far = []
    for i in yuzhi:
        rang_nums, right_nums, dicts, dicts2, = infer(model_, device_, faces_identify, targets, threshold=i,
                                                      tta=False, label=label, names=names)
        # print(rang_nums, right_nums)
        print('阈值-->', i, '正确率-->', right_nums / (rang_nums + right_nums))
        result = Counter(label)
        # print(result)
        # print(dicts, dicts2)
        FRR = 0  # 错误拒绝率 自己不认自己
        index = 0
        FAR = 0  # 错误接受率, 别人认成自己
        for key in dicts.keys():
            FRR += dicts[key] / result[key]
            index += 1
            # print(FRR, index)
        for key in dicts2.keys():
            FAR += dicts2[key]
        if index == 0:
            FRR = 0
        else:
            FRR = FRR / index
        FAR = FAR / (len(targets) * img_nums)
        # print('targets', len(targets), 'img_nums', img_nums)
        print('FRR-->', FRR)
        print('FAR-->', FAR)
        frr.append(FRR * 100)
        far.append(FAR * 100)
        if FAR <= 0.00001:
            print('符合标准的情况', 'FAR->', FAR * 100, 'FRR->', FRR * 100)
            if FRR == 1 or FAR == 0:
                break

    plt.plot(frr, far, 'r-.p', label="plot figure")
    plt.show()
    plt.savefig('./frr-far2.png')
