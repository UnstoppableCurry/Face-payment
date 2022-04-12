import os
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn as nn

# reference:
# https://github.com/ultralytics/yolov3/blob/master/models.py
# https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet/blob/master/yolo/vedanet/network/backbone/brick/darknet53.py
# True 查看相关的网络结构
flag_yolo_structure = False


# 构建CBL模块
class Conv2dBatchLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        '''
        :param in_channels: 输入特征图的通道数
        :param out_channels: 输出特征图的通道数，即卷积核个数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param leaky_slope: leak_relu的系数
        '''
        super(Conv2dBatchLeaky, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
            if flag_yolo_structure:
                print('------------------->>>> Conv2dBatchLeaky isinstance')
        else:
            self.padding = int(kernel_size / 2)

        self.leaky_slope = leaky_slope
        # Layer
        # LeakyReLU : y = max(0, x) + leaky_slope*min(0,x)
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# 构建Resunit模块
class ResBlockSum(nn.Module):
    def __init__(self, nchannels):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels / 2), 1, 1),
            Conv2dBatchLeaky(int(nchannels / 2), nchannels, 3, 1)
        )

    def forward(self, x):
        return x + self.block(x)


# 构建头部分
class HeadBody(nn.Module):
    def __init__(self, in_channels, out_channels):
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


# 上采样
class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


# 网络的输出层，若进行预测返回预测结果
# default anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC):
        print('input anchors', anchors)
        """
        :param anchors:
        :param nC:
        """
        super(YOLOLayer, self).__init__()

        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)  # number of anchors (3)
        self.nC = nC  # number of classes
        self.img_size = 0
        if flag_yolo_structure:
            print('init YOLOLayer ------ >>> ')
            print('anchors  : ', self.anchors)
            print('nA       : ', self.nA)
            print('nC       : ', self.nC)
            print('img_size : ', self.img_size)

    def forward(self, p, img_size, var=None):  # p : feature map
        bs, nG = p.shape[0], p.shape[-1]  # batch_size , grid
        if flag_yolo_structure:
            print('bs, nG --->>> ', bs, nG)
        if self.img_size != img_size:
            create_grids(self, img_size, nG, p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, xywh + confidence + classes)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            io = p.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            io[..., 4:] = torch.sigmoid(io[..., 4:])  # p_conf, p_cls
            io[..., :4] *= self.stride
            if self.nC == 1:
                io[..., 5] = 1  # single-class model
            # flatten prediction, reshape from [bs, nA, nG, nG, nC] to [bs, nA * nG * nG, nC]
            return io.view(bs, -1, 5 + self.nC), p


# 若图像尺寸不是416，调整anchor的生成，输出特征图
def create_grids(self, img_size, nG, device='cpu'):
    # self.nA : len(anchors)  # number of anchors (3)
    # self.nC : nC  # number of classes
    # nG : 输出特征图的大小，与输入图像大小有关
    self.img_size = img_size
    self.stride = img_size / nG
    if flag_yolo_structure:
        print('create_grids stride : ', self.stride)

    # build xy offsets
    grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
    grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4).to(device)
    if flag_yolo_structure:
        print('grid_x : ', grid_x.size(), grid_x)
        print('grid_y : ', grid_y.size(), grid_y)
        print('grid_xy : ', self.grid_xy.size(), self.grid_xy)

    # build wh gains
    print(self.anchors, 'anchors')
    self.anchor_vec = self.anchors.to(device) / self.stride  # 基于 stride 的归一化
    # print('self.anchor_vecself.anchor_vecself.anchor_vec:',self.anchor_vec)
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2).to(device)
    self.nG = torch.FloatTensor([nG]).to(device)


def get_yolo_layer_index(module_list):
    yolo_layer_index = []
    for index, l in enumerate(module_list):
        try:
            a = l[0].img_size and l[0].nG  # only yolo layer need img_size and nG
            yolo_layer_index.append(index)
        except:
            pass
    assert len(yolo_layer_index) > 0, "can not find yolo layer"
    return yolo_layer_index


# ----------------------yolov3------------------------
# 1.1 模型构建
class Yolov3(nn.Module):
    def __init__(self, num_classes=80,
                 anchors=[(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198),
                          (373, 326)]):
        super().__init__()
        # 获取不同输出尺度上的anchor，对于416*416的图像，
        anchor_mask1 = [i for i in range(2 * len(anchors) // 3, len(anchors), 1)]
        anchor_mask2 = [i for i in range(len(anchors) // 3, 2 * len(anchors) // 3, 1)]
        anchor_mask3 = [i for i in range(0, len(anchors) // 3, 1)]

        layer_list = []
        # list0 构建backbone的第一部分，获取52*52的特征图
        layer_list.append(OrderedDict([
            # CBL
            ("0_stage1_conv", Conv2dBatchLeaky(3, 32, 3, 1, 1)),
            # CBL + Resunit
            ("0_stage2_conv", Conv2dBatchLeaky(32, 64, 3, 2)),
            ('0_stage2_ressum1', ResBlockSum(64)),
            # CBL + resunit*2
            ("0_stage3_conv", Conv2dBatchLeaky(64, 128, 3, 2)),
            ('0_stage3_ressum1', ResBlockSum(128)),
            ('0_stage3_ressum2', ResBlockSum(128)),
            # CBL + resunit*8
            ("0_stage4_conv", Conv2dBatchLeaky(128, 256, 3, 2)),
            ('0_stage4_ressum1', ResBlockSum(256)),
            ('0_stage4_ressum2', ResBlockSum(256)),
            ('0_stage4_ressum3', ResBlockSum(256)),
            ('0_stage4_ressum4', ResBlockSum(256)),
            ('0_stage4_ressum5', ResBlockSum(256)),
            ('0_stage4_ressum6', ResBlockSum(256)),
            ('0_stage4_ressum7', ResBlockSum(256)),
            ('0_stage4_ressum8', ResBlockSum(256)),
        ]))

        # list1 构建backbone的第二部分获取26*26的特征图 list1
        layer_list.append(OrderedDict([
            # CBL +resunit*8
            ('1_stage5_conv', Conv2dBatchLeaky(256, 512, 3, 2)),
            ('1_stage5_ressum1', ResBlockSum(512)),
            ('1_stage5_ressum2', ResBlockSum(512)),
            ('1_stage5_ressum3', ResBlockSum(512)),
            ('1_stage5_ressum4', ResBlockSum(512)),
            ('1_stage5_ressum5', ResBlockSum(512)),
            ('1_stage5_ressum6', ResBlockSum(512)),
            ('1_stage5_ressum7', ResBlockSum(512)),
            ('1_stage5_ressum8', ResBlockSum(512)),

        ]))

        # list 2 获取13*13的特征图并输入输出部分
        layer_list.append(OrderedDict([
            # CBL+resunit*4
            ('2_stage6_conv', Conv2dBatchLeaky(512, 1024, 3, 2)),
            ('2_stage6_ressum1', ResBlockSum(1024)),
            ('2_stage6_ressum2', ResBlockSum(1024)),
            ('2_stage6_ressum3', ResBlockSum(1024)),
            ('2_stage6_ressum4', ResBlockSum(1024)),
            # CBL*5
            ('2_headbody1', HeadBody(1024, 512)),
        ]))

        # list 3 获取13*13特征图像的预测 13*13*255
        layer_list.append(OrderedDict([
            ('3_conv_1', Conv2dBatchLeaky(512, 1024, 3, 1)),
            ('3_conv_2', nn.Conv2d(in_channels=1024, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1,
                                   stride=1)),
        ]))

        # list 4 获取13*13特征图上的检测结果 3*((x, y, w, h, confidence) + classes )
        layer_list.append(OrderedDict([
            ('4_yolo', YOLOLayer([anchors[i] for i in anchor_mask1], num_classes)),
        ]))
        # list 5 上采样
        layer_list.append(OrderedDict([
            ('5_conv', Conv2dBatchLeaky(512, 256, 1, 1)),
            ('5_upsample', Upsample(scale_factor=2)),
        ]))

        # list 6 获取26*26的特征图并输入输出部分
        layer_list.append(OrderedDict([
            ('6_head_body2', HeadBody(768, 256)),
        ]))

        # list 7 获取26*26特征图像的预测 26*26*255
        layer_list.append(OrderedDict([
            ('7_conv_1', Conv2dBatchLeaky(256, 512, 3, 1)),
            ('7_conv_2', nn.Conv2d(in_channels=512, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1,
                                   stride=1)),
        ]))

        # list 8 获取26*26特征图上的检测结果 3*((x, y, w, h, confidence) + classes )
        layer_list.append(OrderedDict([
            ('8_yolo', YOLOLayer([anchors[i] for i in anchor_mask2], num_classes)),
        ]))

        # list 9
        layer_list.append(OrderedDict([
            ('9_conv', Conv2dBatchLeaky(256, 128, 1, 1)),
            ('9_upsample', Upsample(scale_factor=2)),
        ]))

        # list 10 获取52*52的特征图并输入输出部分
        layer_list.append(OrderedDict([
            ('10_head_body3', HeadBody(384, 128)),
        ]))

        # list 11 获取52*52特征图像的预测 52*52*255
        layer_list.append(OrderedDict([
            ('11_conv_1', Conv2dBatchLeaky(128, 256, 3, 1)),
            ('11_conv_2', nn.Conv2d(in_channels=256, out_channels=len(anchor_mask1) * (num_classes + 5), kernel_size=1,
                                    stride=1)),
        ]))

        # list 12 获取52*52特征图上的检测结果 3*((x, y, w, h, confidence) + classes )
        layer_list.append(OrderedDict([
            ('12_yolo', YOLOLayer([anchors[i] for i in anchor_mask3], num_classes))
        ]))

        # nn.ModuleList类似于pytho中的list类型，只是将一系列层装入列表
        self.module_list = nn.ModuleList([nn.Sequential(i) for i in layer_list])

        # 获取输出结果 list4 list8 list12
        self.yolo_layer_index = get_yolo_layer_index(self.module_list)

    def forward(self, x):
        # 前向传播
        img_size = x.shape[-1]
        output = []
        # list0
        x = self.module_list[0](x)
        x_route1 = x
        print(x.shape)
        # list1
        x = self.module_list[1](x)
        x_route2 = x
        print(x.shape)
        # list2
        x = self.module_list[2](x)
        # list3
        print(x.shape)
        yolo_head = self.module_list[3](x)
        # list4
        yolo_head_out_13x13 = self.module_list[4][0](yolo_head, img_size)
        output.append(yolo_head_out_13x13)
        # list5
        x = self.module_list[5](x)
        # 融合
        x = torch.cat([x, x_route2], 1)
        # list6
        x = self.module_list[6](x)
        # list7
        yolo_head = self.module_list[7](x)
        # list8
        yolo_head_out_26x26 = self.module_list[8][0](yolo_head, img_size)
        output.append(yolo_head_out_26x26)
        # list9
        x = self.module_list[9](x)
        # 融合
        x = torch.cat([x, x_route1], 1)
        # list10
        x = self.module_list[10](x)
        # list11
        yolo_head = self.module_list[11](x)
        # list12
        yolo_head_out_52x52 = self.module_list[12][0](yolo_head, img_size)
        output.append(yolo_head_out_52x52)
        # 输出结果
        if self.training:
            return output
        else:
            io, p = list(zip(*output))
            return torch.cat(io, 1), p


# 1.2 模型测试
if __name__ == "__main__":
    # 定义模型输入，实例化
    input = torch.Tensor(1, 3, 416, 416)
    model = Yolov3(num_classes=20)
    # 训练阶段
    model.train()
    reses = model(input)
    # for res in reses:
    #     print(np.shape(res))
    # # 预测阶段
    # model.eval()
    # infer_res, train_res = model(input)
    # print('预测', np.shape(infer_res))
    # for res in train_res:
    #     print("训练", np.shape(res))
