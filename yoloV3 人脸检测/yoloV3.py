import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

flag_yolo_structure = False


class Conv2dBatchLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        '''
        CBL模块初始化
        :param in_channels:输入特征图通道
        :param out_channels:输出通道(卷积核个数)
        :param kernel_size:卷积核尺寸
        :param stride:步长
        :param leak_slope:系数
        '''
        super(Conv2dBatchLeaky, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):  # 判断kernelsize 是不是指定类型,(3*3)的元祖或者列表 继承关系也可以,type不考虑继承
            # 也就是算padding的方式而已...
            self.padding = [int(ii / 2) for ii in kernel_size]
            if flag_yolo_structure:
                print('------------------->>>> Conv2dBatchLeaky isinstance', self.padding)
        else:
            self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ResBlockSum(nn.Module):
    def __init__(self, nchannels):
        '''
        残差块,两个CBL+短路
        :param nchannels:
        '''
        super(ResBlockSum, self).__init__()
        self.block = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels / 2), 1, 1),  # 输入通道,输出通道,卷积核尺寸,步长
            Conv2dBatchLeaky(int(nchannels / 2), nchannels, 3, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class HeadBody(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        CBL*5
        :param in_channels:
        :param out_channels:
        '''
        super(HeadBody, self).__init__()
        self.layer = nn.Sequential(
            Conv2dBatchLeaky(in_channels, out_channels, 1, 1),
            Conv2dBatchLeaky(out_channels, out_channels * 2, 3, 1),
            Conv2dBatchLeaky(out_channels * 2, out_channels, 1, 1),
            Conv2dBatchLeaky(out_channels, out_channels * 2, 3, 1),
            Conv2dBatchLeaky(out_channels * 2, out_channels, 1, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, model='nearest'):
        '''
        上采样
        :param scale_factor:
        :param model:
        '''
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = model

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YoloLayer(nn.Module):
    def __init__(self, anchors, nc):
        '''
        网络输出层,进行预测返回结果
        :param anchors:anchors先验眶的尺寸
        :param nc:num_class类别数量
        '''
        super(YoloLayer, self).__init__()
        self.anchors = torch.FloatTensor(anchors)
        self.na = len(anchors)  # 默认一个block3个anchor
        self.nc = nc  # class数量
        self.img_size = 0
        if flag_yolo_structure:
            print('init YOLOLayer ------ >>> ')
            print('anchors  : ', self.anchors)
            print('nA       : ', self.nA)
            print('nC       : ', self.nC)
            print('img_size : ', self.img_size)

    def forward(self, p, img_size, var=None):
        '''
        :param p:batch-size,3*85,13,13  特征图
        :param img_size:原图尺寸
        :param var:
        :return:
        '''
        bs, nG = p.shape[0], p.shape[-1]  # batch_size, grid 第几个网格
        if flag_yolo_structure:
            print('bs, nG --->>> ', bs, nG)
        if self.img_size != img_size:
            create_grids(self, img_size, nG, p.device)
        p = p.view(bs, self.na, self.nc + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        # tensor view contiguous等操作后底层内存是共享的,所以contiguous要创建一个新的内存空间存储新的变量
        if self.training:
            # 如果是训练模式 直接返回P也就是 yolo的 目标值
            return p
        else:
            io = p.clone()
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # 中心点坐标+偏移量(左上角距离)
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # 宽高归一化后 乘 目标框宽高~
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # 置信度
            io[..., :4] *= self.stride  # xywh参数 乘 缩放系数 逆一化
            if self.nc == 1:
                io[..., 5] = 1  # 如果是训练模式,label置信度设置为1
            # batchsize ,    有几个grid算几个grid 自动匹配呗,类别:80 + 4 xywh + 置信度标签
            return io.view(bs, -1, 5 + self.nc), p


def create_grids(self, img_size, num_grid, device='cpu'):
    '''
    创建目标框
    :param img_size:输入尺寸
    :param num_grid:输出特征图尺寸
    :param device:驱动
    :return:
    '''
    self.img_size = img_size
    self.stride = img_size / num_grid  # 缩放倍数
    if flag_yolo_structure:
        print('create_grids stride : ', self.stride)
    grid_x = torch.arange(num_grid).repeat((num_grid, 1)).view((1, 1, num_grid, num_grid)).float()
    # arange 数量上少一个 类似range 从0-numgrid数组生成
    # repeat是复制,复制13行1列...不知道是干啥
    # view 将数据按照顺序填充
    grid_y = grid_x.permute(0, 1, 3, 2)  # 2-3交换位置 正好代表的含义是行转列
    self.grid_xy = torch.stack((grid_x, grid_y), 4).to(device)  # 将x,y信息cat在一起构成特征图
    # 直接计算每个anchor的偏移量 因为是归一化的,所以直接计算倍数即可
    # concat是纵轴拼接维度不变,stack是横着拼在指定维度再添加一个维度
    if flag_yolo_structure:
        print('grid_x : ', grid_x.size(), grid_x)
        print('grid_y : ', grid_y.size(), grid_y)
        print('grid_xy : ', self.grid_xy.size(), self.grid_xy)
    # print(self.anchors, self.anchors.shape)
    self.anchor_vec = self.anchors.to(device) / self.stride  # 基于缩放倍数的归一化也就是将anchors缩放同等比例
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device)  # 宽高是归一化的,直接计算每个anchor框所对应的缩放比例即可
    self.ng = torch.FloatTensor([num_grid]).to(device)


def get_yolo_layer_index(module_list):
    yolo_layer_index = []
    for index, l in enumerate(module_list):
        try:
            a = l[0].img_size and l[0].ng
            yolo_layer_index.append(index)
        except:
            pass
    assert len(yolo_layer_index) > 0, '找不到yolo输出层'
    return yolo_layer_index


anch = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198),
        (373, 326)]


class Yolov3(nn.Module):
    def __init__(self, num_classes=80, anchors=anch
                 ):
        super(Yolov3, self).__init__()
        # 获取不同尺度上的anchor 先验框 对于416*416
        anchor_mask1 = [i for i in range(2 * len(anchors) // 3, len(anchors), 1)]  # 大
        anchor_mask2 = [i for i in range(len(anchors) // 3, len(anchors) * 2 // 3, 1)]  # 中
        anchor_mask3 = [i for i in range(0, len(anchors) // 3, 1)]  # 小

        layer_list = []  # list 用来构建backbone的第一部分 获取52*52x256的特征图
        layer_list.append(OrderedDict(
            [
                # CBL
                ('0_stage1_conv', Conv2dBatchLeaky(3, 32, 3, 1, 1)),
                # CBL +Resunit
                ("0_stage2_conv", Conv2dBatchLeaky(32, 64, 3, 2)),
                ('0_stage2_resnetsum', ResBlockSum(64)),
                # CBL+ resunit
                ("0_stage3_conv", Conv2dBatchLeaky(64, 128, 3, 2)),
                ("0_stage3_resnetsum1", ResBlockSum(128)),
                ("0_stage3_resnetsum2", ResBlockSum(128)),
                # CBL + resunit*8
                ("0_stage4_conv", Conv2dBatchLeaky(128, 256, 3, 2)),
                ("0_stage4_resnetsum1", ResBlockSum(256)),
                ("0_stage4_resnetsum2", ResBlockSum(256)),
                ("0_stage4_resnetsum3", ResBlockSum(256)),
                ("0_stage4_resnetsum4", ResBlockSum(256)),
                ("0_stage4_resnetsum5", ResBlockSum(256)),
                ("0_stage4_resnetsum6", ResBlockSum(256)),
                ("0_stage4_resnetsum7", ResBlockSum(256)),
                ("0_stage4_resnetsum8", ResBlockSum(256)),

            ]
        ))
        # list2 构建backbone第二部分 获取 26*26x512的特征图
        layer_list.append(OrderedDict([
            # CBL + resunit*
            ("1_stage5_conv", Conv2dBatchLeaky(256, 512, 3, 2)),
            ("1_stage5_resnetsum1", ResBlockSum(512)),
            ("1_stage5_resnetsum2", ResBlockSum(512)),
            ("1_stage5_resnetsum3", ResBlockSum(512)),
            ("1_stage5_resnetsum4", ResBlockSum(512)),
            ("1_stage5_resnetsum5", ResBlockSum(512)),
            ("1_stage5_resnetsum6", ResBlockSum(512)),
            ("1_stage5_resnetsum7", ResBlockSum(512)),
            ("1_stage5_resnetsum8", ResBlockSum(512)),
        ]))
        # list 3 获取13*13x512特征图  然后输入输出部分 FPN
        layer_list.append(OrderedDict([
            # CBL+resunit*4
            ("2_stage6_conv", Conv2dBatchLeaky(512, 1024, 3, 2)),
            ('2_stage6_ressum1', ResBlockSum(1024)),
            ('2_stage6_ressum2', ResBlockSum(1024)),
            ('2_stage6_ressum3', ResBlockSum(1024)),
            ('2_stage6_ressum4', ResBlockSum(1024)),
            # CBL*5
            ('2_headbody1', HeadBody(1024, 512)),
        ]))
        # list4 获取13x13特征图的预测 13x13x255
        layer_list.append(OrderedDict([
            ("3_conv_1", Conv2dBatchLeaky(512, 1024, 3, 1)),
            ('3_conv_2',
             nn.Conv2d(in_channels=1024, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1, stride=1))
        ]))
        # layer5 获取13x13特征图的检测结果
        layer_list.append(OrderedDict([
            ('4_yolo', YoloLayer([anchors[i] for i in anchor_mask1], num_classes)),
        ]))
        # list6 上采样
        layer_list.append(OrderedDict([
            ('5_conv', Conv2dBatchLeaky(512, 256, 1, 1)),
            ('5_upsample', Upsample(scale_factor=2)),
        ]))
        # list7 获取26x26的特征图并输入输出部分
        layer_list.append(OrderedDict([
            ('6_head_body2', HeadBody(768, 256)),  # 256+512 13x13上采样concat 26x26x512
        ]))
        # list8  获取26x26特征图的预测 26x26x255
        layer_list.append(OrderedDict([
            ('7_conv_1', Conv2dBatchLeaky(256, 512, 3, 1)),
            ('7_conv2',
             nn.Conv2d(in_channels=512, out_channels=len(anchor_mask2) * (num_classes + 5), kernel_size=1, stride=1))
        ]))
        # list9 获取 26x26检测结果
        layer_list.append(OrderedDict([
            ('8_yolo', YoloLayer([anchors[i] for i in anchor_mask2], num_classes)),
        ]))
        # list 10  26x26x256 上采样
        layer_list.append(OrderedDict([
            ('9_conv', Conv2dBatchLeaky(256, 128, 1, 1)),
            ('9_upsample', Upsample(scale_factor=2)),  # 2倍下采样
        ]))
        # list11 获取52x52特征图输入FPN部分
        layer_list.append(OrderedDict([
            ('10_head_body3', HeadBody(384, 128)),
        ]))
        # list12 获取52x52x255特征图的预测
        layer_list.append(OrderedDict([
            ('11_conv_1', Conv2dBatchLeaky(128, 256, 3, 1)),
            ('11_conv_2',
             nn.Conv2d(in_channels=256, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1, stride=1))
        ]))
        # list 13 获取52x52特征图检测结果
        layer_list.append(OrderedDict([
            ('12_yolo', YoloLayer([anchors[i] for i in anchor_mask3], num_classes))
        ]))
        # nn.ModuleList类似于 list类型 只是封装进去而已
        self.module_list = nn.ModuleList([nn.Sequential(i) for i in layer_list])
        # 获取输出结果 list5,list9,list13
        self.yolo_layer_index = get_yolo_layer_index(self.module_list)

    def forward(self, x):
        img_size = x.shape[-1]  # batch_size,通道数量=3,尺寸=416x416
        output = []
        # 1-9层 list1 计算
        x = self.module_list[0](x)
        x_cach1 = x  # 52x52x256
        # 10-14 list2
        x = self.module_list[1](x)
        x_cach2 = x  # 26x26x512
        # 14-15 list3
        x = self.module_list[2](x)  # 13x13x512
        # 16   list4
        yolo_head = self.module_list[3](x)  # backbone输出13x13x512特征图
        # list 5
        yolo_head_out_13x13 = self.module_list[4][0](yolo_head, img_size)
        output.append(yolo_head_out_13x13)  # 13x13x255
        # list 6
        x = self.module_list[5](x)
        # 13x13x512->13x13x256 up-> 26x26x256 concat 26x26x512
        # concat
        x = torch.cat([x, x_cach2], 1)
        # list7  26x26x768->26x26x256
        x = self.module_list[6](x)
        # list8 26x26x256->26x26x512->26x26x(3x(num_class+5))
        yolo_head = self.module_list[7](x)
        # list9 预测输出预测框+置信度 x 3
        yolo_head_out_26x26 = self.module_list[8][0](yolo_head, img_size)
        output.append(yolo_head_out_26x26)
        # list10 26x26x256->26x26x128->up 52x52x128
        x = self.module_list[9](x)
        # concat 52x52x256 - 52x52x128->52x52x384
        x = torch.cat([x, x_cach1], 1)
        # list11 52x52x384->52x52x128
        x = self.module_list[10](x)
        # list12 52x52x128->52x52x256->52x52x(3*(num_class+5))
        yolo_head = self.module_list[11](x)
        # 预测结果
        yolo_head_out_52x52 = self.module_list[12][0](yolo_head, img_size)
        output.append(yolo_head_out_52x52)
        # 输出结果
        if self.training:
            return output
        else:
            io, p = list(zip(*output))
            return torch.cat(io, 1), p  # io 是 x,y,w,h,+置信度+类别one-hot,p是特征图 用预训练时计算损失


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Yolov3Tiny(nn.Module):
    def __init__(self, num_classes=80, anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]):
        super(Yolov3Tiny, self).__init__()
        # 6个anchor,输出尺度有两种，还是32倍下采样
        # anchor_mask1是13*13 大物体
        # anchor_mask2是26*26 中等物体
        anchor_mask1 = [i for i in range(len(anchors) // 2, len(anchors), 1)]  # [3, 4, 5]
        anchor_mask2 = [i for i in range(0, len(anchors) // 2, 1)]  # [0, 1, 2]
        # 网络构建，所有的网络层都存放在layerlist中，
        # OrderedDict 是 dict 的子类，其最大特征是可以保持添加的key-valu对的顺序
        # 直接按照网络的构成顺序构建网络
        layer_list = []
        layer_list.append(OrderedDict([
            # layer 0
            ("conv_0", nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_0", nn.BatchNorm2d(16)),
            ("leaky_0", nn.LeakyReLU(0.1)),
            # layer 1
            ("maxpool_1", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 2
            ("conv_2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_2", nn.BatchNorm2d(32)),
            ("leaky_2", nn.LeakyReLU(0.1)),
            # layer 3
            ("maxpool_3", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 4
            ("conv_4", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_4", nn.BatchNorm2d(64)),
            ("leaky_4", nn.LeakyReLU(0.1)),
            # layer 5
            ("maxpool_5", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 6
            ("conv_6", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_6", nn.BatchNorm2d(128)),
            ("leaky_6", nn.LeakyReLU(0.1)),
            # layer 7
            ("maxpool_7", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 8
            ("conv_8", nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_8", nn.BatchNorm2d(256)),
            ("leaky_8", nn.LeakyReLU(0.1)),
        ]))

        layer_list.append(OrderedDict([
            # layer 9
            ("maxpool_9", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
            # layer 10
            ("conv_10", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_10", nn.BatchNorm2d(512)),
            ("leaky_10", nn.LeakyReLU(0.1)),
            # layer 11
            ('_debug_padding_11', nn.ZeroPad2d((0, 1, 0, 1))),
            ("maxpool_11", nn.MaxPool2d(kernel_size=2, stride=1, padding=0)),
            # layer 12
            ("conv_12", nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_12", nn.BatchNorm2d(1024)),
            ("leaky_12", nn.LeakyReLU(0.1)),
            # layer 13
            ("conv_13", nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)),
            ("batch_norm_13", nn.BatchNorm2d(256)),
            ("leaky_13", nn.LeakyReLU(0.1)),
        ]))

        layer_list.append(OrderedDict([
            # layer 14
            ("conv_14", nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_14", nn.BatchNorm2d(512)),
            ("leaky_14", nn.LeakyReLU(0.1)),
            # layer 15
            ("conv_15",
             nn.Conv2d(in_channels=512, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1, stride=1,
                       padding=0, bias=True)),
        ]))

        # layer 16 13*13特征图检测的结果
        anchor_tmp1 = [anchors[i] for i in anchor_mask1]
        layer_list.append(OrderedDict([("yolo_16", YoloLayer(anchor_tmp1, num_classes))]))

        # layer 17
        layer_list.append(OrderedDict([("route_17", EmptyLayer())]))

        layer_list.append(OrderedDict([
            # layer 18
            ("conv_18", nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)),
            ("batch_norm_18", nn.BatchNorm2d(128)),
            ("leaky_18", nn.LeakyReLU(0.1)),
            # layer 19
            ("upsample_19", Upsample(scale_factor=2)),
        ]))

        # layer 20
        layer_list.append(OrderedDict([('route_20', EmptyLayer())]))

        layer_list.append(OrderedDict([
            # layer 21
            ("conv_21", nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
            ("batch_norm_21", nn.BatchNorm2d(256)),
            ("leaky_21", nn.LeakyReLU(0.1)),
            # layer 22
            ("conv_22",
             nn.Conv2d(in_channels=256, out_channels=len(anchor_mask2) * (num_classes + 5), kernel_size=1, stride=1,
                       padding=0, bias=True)),
        ]))

        # layer 23 26*26特征图的输出结果
        anchor_tmp2 = [anchors[i] for i in anchor_mask2]
        layer_list.append(OrderedDict([("yolo_23", YoloLayer(anchor_tmp2, num_classes))]))
        # nn.ModuleList类似于pytho中的list类型，只是将一系列层装入列表
        self.module_list = nn.ModuleList([nn.Sequential(layer) for layer in layer_list])
        # 网络的输出层
        self.yolo_layer_index = get_yolo_layer_index(self.module_list)

    def forward(self, x):
        # 前向传播过程
        img_size = x.shape[-1]
        output = []
        # layer0 to layer8
        x = self.module_list[0](x)
        x_route8 = x
        # layer9 to layer13
        x = self.module_list[1](x)
        x_route13 = x
        # layer14, layer15
        x = self.module_list[2](x)
        # yolo_16 13*13特征图的输出
        x = self.module_list[3][0](x, img_size)
        output.append(x)
        # layer18, layer19
        x = self.module_list[5](x_route13)
        # 特征融合
        x = torch.cat([x, x_route8], 1)
        # layer21, layer22
        x = self.module_list[7](x)
        # yolo_23 26*26特征图的输出
        x = self.module_list[8][0](x, img_size)
        output.append(x)
        # 训练是直接输出两个尺度的结果，
        # train_out: torch.Size([5, 3, 13, 13, 85])
        # train_out: torch.Size([5, 3, 26, 26, 85])
        # 预测时进行拼接
        # inference_out: torch.Size([5, 2535, 85])
        if self.training:
            return output
        else:
            io, p = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), p


if __name__ == '__main__':
    input = torch.Tensor(5, 3, 608, 608)
    model = Yolov3Tiny(num_classes=20)
    model.train()
    reses = model(input)
    for res in reses:
        print(np.shape(res))
    # 预测阶段
    model.eval()
    a, b = model(input)
    print('预测', np.shape(a))
    for res in b:
        print("训练")
