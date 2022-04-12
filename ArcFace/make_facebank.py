import warnings

warnings.filterwarnings('ignore')
import os
import torch
from model import Backbone
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import io
from torchvision import transforms as trans
import torch
from model import l2_norm


def prepare_facebank(path_images, facebank_path, model, device, tta=True):
    '''
    创建人脸数据特征向量
    :param path_images:图像路径
    :param facebank_path:输出保存路径
    :param model:人脸特征提取使用的模型
    :param device:设备信息
    :param tta:是否获取镜像的特征
    :return:
    '''
    test_transform_ = trans.Compose([
        trans.RandomResizedCrop(112, scale=(1.0, 1.0), ratio=(1.0, 1.0)),  # scale随机裁剪的面积占比,ratio随机裁剪长宽比
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    model.eval()
    embeddings = []
    names = ['Unknown']
    idx = 0  # 人脸类别数
    for path in path_images.iterdir():
        if path.is_file():
            continue
        else:
            idx += 1
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file).convert('RGB')  # 读取图像
                        print(" {}) {}".format(idx + 1, file))
                    except:
                        continue
                    if img.size != (112, 112):
                        try:
                            img = img.resize((112, 112))
                        except:
                            continue
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(test_transform_(img).to(device).unsqueeze(0))
                            emb_mirror = model(test_transform_(mirror).to(device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:
                            embs.append(model(test_transform_(img).to(device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)  # shape(cls_num.1,embedding)->shape(cls_num,embedding)
    names = np.array(names)
    torch.save(embeddings, facebank_path + '/facebank_1.pth')
    np.save(facebank_path + '/names_1', names)
    return embeddings, names


if __name__ == '__main__':
    # 参数配置
    parser = argparse.ArgumentParser(description='make facebank')
    # 模型
    parser.add_argument("--net_mode", help="which network, [ir, ir_se, mobilefacenet]", default='ir_se', type=str)
    # 网络深度
    parser.add_argument("--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    # 预训练模型
    parser.add_argument("--finetune_backbone_model", help="finetune_backbone_model",
                        # default="/root/cv/pycharm/人脸检测/人脸识别/save/model_2022-01-31-02-22-13_step_19188.pth",
                        # default="/root/cv/pycharm/人脸检测/人脸识别/save/model_2022-01-31-05-27-40_step_2144.pth",
                        default="/root/cv/pycharm/人脸检测/人脸识别/save/model_2022-01-31-12-05-37_step_12864.pth", #自己训练标准训练集99epoch
                        # default="/root/cv/pycharm/人脸检测/人脸识别/local_save/model_2022-02-02-05-23-17_step_169.pth",
                        # default="/root/cv/pycharm/人脸检测/人脸识别/face_verify-model_ir_se-50.pth",  # 老师训练模型
                        # default="./local_save/model_2022-01-31-07-27-53_step_400.pth",
                        type=str)
    # 人脸仓库中的人脸图像
    parser.add_argument("--facebank_images_path", help="facebank_images_path",
                        # default="C:\\Users\86183\Desktop\人脸识别数据集\imgs\\", type=str)
                        # default="/root/cv/dataset/人脸/datasets/insight_face/imgs", type=str)
                        default="/root/cv/pycharm/人脸检测/人脸识别/mydataset2", type=str)

    # 人脸仓库
    parser.add_argument("--facebank_path", help="facebank_path", default="./facebank/", type=str)
    # 是否翻转
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", default=False, type=bool)

    args = parser.parse_args()
    # 设备信息
    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型选择
    model_ = Backbone(args.net_depth, 1., args.net_mode).to(device_)
    print('{}_{} model generated'.format(args.net_mode, args.net_depth))
    # 加载预训练模型
    if os.access(args.finetune_backbone_model, os.F_OK):
        model_.load_state_dict(torch.load(args.finetune_backbone_model, map_location='cpu'))
        print("-------->>>   load model : {}".format(args.finetune_backbone_model))
    # 模型预测
    model_.eval()
    # 创建模型仓库
    targets, names = prepare_facebank(Path(args.facebank_images_path), args.facebank_path, model_, device_,
                                      tta=args.tta)
