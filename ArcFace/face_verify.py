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


# 加载pth,npy文件中存储的特征
def load_facebank(facebank_path):
    embeddings = torch.load(facebank_path + '/facebank.pth',
                            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    names = np.load(facebank_path + '/names.npy')
    return embeddings, names


def infer(model, device, faces, target_embs, threshold=1.2, tta=False):
    '''
    :param model: 进行预测的模型
    :param device: 设备信息
    :param faces: 要处理的人脸图像
    :param target_embs: 数据库中的人脸特征
    :param threshold: 阈值
    :param tta: 进行水平翻转的增强
    :return:
    '''
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
    diff_1 = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
    dist_1 = torch.sum(torch.pow(diff_1, 2), dim=1)

    diff = torch.mm(source_embs, target_embs.transpose(1, 0))
    dist = torch.pow(diff, 2) * 64

    # print('dist_1-->', dist_1)
    # print('dist-->', dist)

    # 获取差异最小值及对应的索引
    # minimum, min_idx = torch.min(dist.squeeze(), dim=0) #计算欧式距离 距离越小说明相似度越高
    minimum, min_idx = torch.max(dist.squeeze(), dim=0)  # 计算cos(x)时,值越大说明夹角越小
    # 若没有匹配成功，将索引设置为-1
    # min_idx[minimum > threshold] = -1
    min_idx[minimum < threshold] = -1

    return min_idx, minimum.unsqueeze(0)


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
                        default="/root/cv/pycharm/人脸检测/人脸识别/save/model_2022-01-31-12-05-37_step_12864.pth", #自己训练标准训练集99epoch
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
    idx = 0
    # 遍历要处理的图像
    for file in os.listdir(args.example):
        # 若非图片文件，进行下一次循环
        # if not file.endswith('png'):
        #     continue
        # 读取图像数据
        img = cv2.imread(args.example + file)
        label = file.split('-')[0]
        if img is None:
            continue
        # 获取图像的宽高
        x, y = img.shape[0:2]
        # 送入网络中的图像必须是112*112
        if x != 112 or y != 112:
            img = cv2.resize(img, (112, 112))
        # 将数据放入list中
        faces_identify.append(Image.fromarray(img))
        # 进行检测，results是索引，face_dst是差异
        results, face_dst = infer(model_, device_, faces_identify, targets, threshold=2.7, tta=False)
        faces_identify.pop()
        # 将其转换numpy的格式
        face_dst = list(face_dst.cpu().detach().numpy())
        # 获取姓名和差异的大小
        print("{}) recognize：{} ,dst : {}".format(idx + 1, names[results + 1], face_dst))
        # 将检测结果绘制在图像上
        # cv2.putText(img, names[results[idx] + 1], (2, 13), cv2.FONT_HERSHEY_DUPLEX, 0.5, (55, 0, 220), 5)
        cv2.putText(img,
                    label + '-' + names[results + 1], (2, 13),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 50, 50), 1)
        plt.imshow(img[:, :, ::-1])
        plt.show()
        # cv2.namedWindow("imag_face", 0)
        # cv2.imshow("imag_face", img)
        # cv2.waitKey(1)
        # 将结果写入到文件中
        # cv2.imwrite(args.example + "results/" + file, img)
        idx += 1
        cv2.destroyAllWindows()
        print()
        print('------------------------')
