import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


def load_model(model, pretrained_state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if
                       k in model_dict and model_dict[k].size() == pretrained_state_dict[k].size()}
    model.load_state_dict(pretrained_dict, strict=False)
    if len(pretrained_dict) == 0:
        print('尚未加载参数')
    else:
        for k, v in pretrained_state_dict.items():
            if k in pretrained_dict:
                print('->load{}  {}'.format(k, v.size()))
            else:
                print('[info] skip {}   {}'.format(k, v.size()))
    return model


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    # 每一个残差块中的channel都是恒定的 所以倍数是1
    expansion = 1

    def __init__(self, inpulanes, planes, stride=1, downsample=None):
        '''
        基础模块初始化
        :param inpulanes: 输入
        :param planes: 卷积
        :param stride: 步长
        :param downsample:下采样倍数,用于模块与模块之间连接时尺度的计算
        '''
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inpulanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)  # 节省内存操作
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # CBL
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # CB
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 瓶颈模块
class Bottleneck(nn.Module):
    expansion = 4

    # 每一个残差块中最后一个卷积核都会扩大
    # conv2x中 前两个通道数量是64 而最后一个卷积层的通道数是256channel
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        '''
        瓶颈模块初始化用于 50-152restnet构建 降低参数量
        :param inplanes:输入尺寸
        :param planes:输出尺寸
        :param stride:步长
        :param downsample:下采样
        '''
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # 1x1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1x1
        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, img_size=224, dropout_factor=1.):
        '''
        :param block:残差结构
        :param layers:残差结构的具体数量,例如res50[3,4,6,3]==16x3 +2 =50
        :param num_classes:网络输出的类别数量
        :param img_size:图像大小
        :param droput_factor:随机失活的概率
        '''
        self.inplanes = 64  # channel定义为64 是因为maxpool后的channel变为64
        self.dropout_factor = dropout_factor
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 3通道 64维度卷积核 卷积尺寸7x7 步长2 填充3
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 构建conv2x,3x,4x,5x部分
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    ceil_mode=True)  # ceil_mode 计算方式,如果数据不够填充 仍然进行剩余计算最大值 如果设置为False 则只计算够的相当于是否截断操作

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 64 * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64 * 8, layers[3], stride=2)
        # fc
        assert img_size % 32 == 0  # 整个网络是32倍下采样,所以图像要允许可以被32整除
        pool_kernel = int(img_size / 32)
        self.avgpool = nn.AvgPool2d(pool_kernel, stride=1, ceil_mode=True)
        # 全局平均池化 (n,c,1,1)的特征图
        self.dropout = nn.Dropout(self.dropout_factor)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # res18/32 fc 维度512,res50/101/152 fc 维度2048
        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 卷积核尺寸x 输出维度
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 根号下 矩阵节点数量分之一
                # 卷积层初始化
            elif isinstance(m,nn.BatchNorm2d):  # bn层初始化
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 1x1下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 第一个残差块通道数发生改变,修正
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 残差模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 全局池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # flatten展平
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 2.4 构建不同层的网络
def resnet18(pretrained=False, **kwargs):
    # 模型初始化
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # 加载预训练模型
        print("Load pretrained model from {}".format(model_urls['resnet18']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        model = load_model(model, pretrained_state_dict)
    return model


def resnet34(pretrained=False, **kwargs):
    # 模型初始化
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # 加载预训练模型
        print("Load pretrained model from {}".format(model_urls['resnet34']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        model = load_model(model, pretrained_state_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    # 模型初始化
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # 加载预训练模型
        print("Load pretrained model from {}".format(model_urls['resnet50']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        model = load_model(model, pretrained_state_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    # 模型初始化
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # 加载预训练模型
        print("Load pretrained model from {}".format(model_urls['resnet101']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        model = load_model(model, pretrained_state_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    # 模型初始化
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        # 加载预训练模型
        print("Load pretrained model from {}".format(model_urls['resnet152']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        model = load_model(model, pretrained_state_dict)
    return model

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 3.模型测试
if __name__ == "__main__":
    model = resnet34(False, num_classes=3, img_size=224)
    input = torch.randn(32, 3, 224, 224)
    output = model(input)
    print(output.size())
