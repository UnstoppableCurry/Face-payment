from pathlib import Path
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import pickle
import torch
from tqdm import tqdm


def get_train_dataset(imgs_folder):
    # 水平翻转,标准化
    train_transform = trans.Compose(
        [
            trans.RandomResizedCrop(112, scale=(1.0, 1.0), ratio=(1.0, 1.0)),  # scale随机裁剪的面积占比,ratio随机裁剪长宽比
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    # 一个通用的数据加载器默认数据及已经按照分配的类型分成了不同的文件夹
    # 一种类型的文件架下面只存放一种类型的图片
    ds = ImageFolder(imgs_folder, train_transform)
    # ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
    # root 指定路径
    # transform 对PIL IMAGE 进行的转换操作 transform的输入是使用loader读取图片返回的对象
    # target_tf 对标签label的转换
    # loader 给定路径后如何读取图片,默认读取RGB格式的PIL Image对象
    # lass_num类别个数
    class_num = len(ds.classes)
    return ds, class_num


def get_train_loader(conf):
    '''
    加载训练集
    :param conf:
    :return:
    '''
    ds, class_num = get_train_dataset(conf.datasets_train_path + '/imgs')
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True)
    return loader, class_num, ds.__len__()


# 反归一化
def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


if __name__ == "__main__":
    from config import get_config
    import matplotlib.pyplot as plt

    # 获取参数配置信息
    conf = get_config()
    # 设置数据的路径
    conf.datasets_train_path = "/root/cv/dataset/人脸/datasets/insight_face"
    # 获取送入网络中的数据
    data_loader, class_num, datasets_len = get_train_loader(conf)
    # 打印数据数量和类别个数
    print("train datasets len : {}".format(datasets_len))
    print(" class_num:{} ".format(class_num))
    # 遍历数据进行展示
    for i, (imgs, labels) in enumerate(data_loader):
        # 遍历每个batch中的每一副图像进行展示
        for j in range(conf.batch_size):
            # 展示
            # cv2.imshow('results', np.uint8(de_preprocess(imgs[j].permute(1, 2, 0)) * 255.0)[:, :, ::-1])
            # cv2.waitKey(0)
            plt.imshow(np.uint8(de_preprocess(imgs[j].permute(1, 2, 0)) * 255.0)[:, :, ::-1])
            plt.show()
            # 打印相应的目标值
            print(labels[j])
    cv2.destroyAllWindows()
