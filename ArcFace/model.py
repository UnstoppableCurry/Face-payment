from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb


class Flatten(Module):
    def forward(self, input):
        '''
        拍平
        :param input:
        :return:
        '''
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    '''
    l2 正则范数
    :param input:
    :param axis:
    :return:
    '''
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        '''
        :param channels:输入通道数
        :param reduction:缩放比例
        '''
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)  # 1x1卷积实现 将通道融合降维,然后再还原回去
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x  # 输入直接×权重


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        '''
        改进型瓶IR残差块,新增BN层 与relu激活的优化 步长的调整,一个是为了降低参数量的初衷,一个是希望步长为1时更好的提取特征图
        :param in_channel:输入通道数
        :param depth:输出通道数
        :param stride:步长
        '''
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            # 短连接 输入通道数与输出通道数相同 直接使用池化操作,池化操作取决于步长
            # 步长为1 不变
            # 步长为2 下采样2倍,对应了残差块第二个卷积核步长为2的情况的尺度变换
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            # 不相等的情况就需要使用1x1卷积核 去匹配指定维度
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        # 残差块部分
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),  # 卷积核尺寸3x3 步长1x1 填充padding=1
            PReLU(depth),
            Conv2d(depth, in_channel, (3, 3), stride, 1, bias=False),  # 卷积核尺寸3x3 步长stride:论文中为2 填充padding=1
            BatchNorm2d(depth)  # 还原回来的尺度
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        '''
        瓶颈模块中天剑SE模块
        :param in_channel:输入通道
        :param depth: 输出通道
        :param stride: 步长
        '''
        super(bottleneck_IR_SE, self).__init__()
        # 短连接部分
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        # 残差部分加入se模块
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


# 定义元祖类型类
class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=64 * 2, num_units=4),
            get_block(in_channel=64 * 2, depth=64 * 4, num_units=14),
            get_block(in_channel=64 * 4, depth=64 * 8, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=64 * 2, num_units=13),
            get_block(in_channel=64 * 2, depth=64 * 4, num_units=30),
            get_block(in_channel=64 * 4, depth=64 * 8, num_units=3),

        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=64 * 2, num_units=8),
            get_block(in_channel=64 * 2, depth=64 * 4, num_units=36),
            get_block(in_channel=64 * 4, depth=64 * 8, num_units=3),
        ]
    return blocks


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        '''
        骨干网络
        :param num_layers:网路层数
        :param drop_ratio:随机失活概率
        :param mode:是否添加se模块
        '''
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layer 必须是50,100,152其中之一,其他层数尚未支持'
        assert mode in ['ir', 'ir_se'], 'model 必须为ir 或者ir_se,其他尚未支持'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        # 输入层
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64)
        )
        # 输出层
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(drop_ratio),
            Flatten(),
            Linear(512 * 7 * 7, 512),  # restnet 输出...7*7*512 展平向量
            BatchNorm1d(512)
        )
        # 残差模块部分
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride
                    )
                )
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


class Arcface(Module):
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        '''
        :param embedding_size:人脸图像特征向量
        :param classnum:人脸分类数,人的个数
        :param s:半径
        :param m:夹角差值
        '''
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul(1e5)  # w权重初始化
        self.m = m  # m2夹角差值,默认0.5
        self.s = s  # 缩放倍数 默认64
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m
        # 阈值 防止角度超过 π
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label):
        # 权重规范化
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # 将特征向量与权重相乘 获取cos相似度
        cos_theta = torch.mm(embbedings, kernel_norm)  # 1 x nB X nB x cls
        cos_theta = cos_theta.clamp(-1, 1)  # 将数值规范化在 -1 ,1 之间 更加稳定
        cos_theta_2 = torch.pow(cos_theta, 2)  # 求平方
        sin_theta_2 = 1 - cos_theta_2  # sin(x)=1-cos(x)^2
        sin_theta = torch.sqrt(sin_theta_2)
        # cos_theta_m 是相似度向量添加类似l2惩罚项, 仅对目标类别生效
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)  # cos(x+m)=cos(x) cos(m)- sin(x)sin(m)
        cond_v = cos_theta - self.threshold  # cos(x)-m
        cond_mask = cond_v <= 0  # 所有超出阈值的类别会被标记为True
        # 大于0 说明theta在 m的左边,小于0说明theta在m的右边
        # 在右边说明超出阈值,非法,进行正则惩罚
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]  # 将所有需要替换的类别使用m3惩罚项替换
        output = cos_theta * 1.0  # cos_theta 是目标类别与每个类别相似度向量
        # 获取类别索引值
        idx_ = torch.arange(0, nB, dtype=torch.long)  # batch 中下标 而不是batch=1时数据类别的下标
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s
        return output
