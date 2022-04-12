import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

# 使用pytorch官方提供的预训练模型
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# 定义一个3*3的卷积，padding为1
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# 定义残差模块
class BasicBlock(nn.Module):
    # 每一个残差块中的channel都是恒定的，所以倍数是1
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # downsample对应的是ResNet网络结构中的虚线连接
        # 比如说 ResNet34中的conv2_x到conv3_x的过渡，
        # 在shortcut分支需要使用虚线连接，从而将 56*56*64 的输入特征矩阵转变为 28*28*128 的输出特征
        super(BasicBlock, self).__init__()
        # 卷积
        self.conv1 = conv3x3(inplanes, planes, stride)
        # BN层
        self.bn1 = nn.BatchNorm2d(planes)
        # 激活
        self.relu = nn.ReLU(inplace=True)
        # 卷积
        self.conv2 = conv3x3(planes, planes)
        # BN层
        self.bn2 = nn.BatchNorm2d(planes)
        # 下采样倍数
        self.downsample = downsample
        # 步长
        self.stride = stride

    def forward(self, x):
        # 短连接部分
        residual = x
        # CBL
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # CB
        out = self.conv2(out)
        out = self.bn2(out)
        # 降采样
        if self.downsample is not None:
            residual = self.downsample(x)
        # 求和
        out += residual
        # 激活
        out = self.relu(out)
        # 输出结果
        return out


# 定义瓶颈模块
class Bottleneck(nn.Module):
    # 每一个残差块中最后一个卷积核都会扩大的倍数是4
    # 比如conv2_x中前两个是64 channel，而最后一个卷积层是 256 channel，
    # 所以在Bottleneck中的expansion为4
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1*1 卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # BN层
        self.bn1 = nn.BatchNorm2d(planes)
        # 3*3卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # BN层
        self.bn2 = nn.BatchNorm2d(planes)
        # 1*1卷积
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # BN层
        self.bn3 = nn.BatchNorm2d(planes * 4)
        # 激活
        self.relu = nn.ReLU(inplace=True)
        # 下采样
        self.downsample = downsample
        # 步长
        self.stride = stride

    def forward(self, x):
        # 短连接
        residual = x
        # 1*1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3*3卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1*1卷积
        out = self.conv3(out)
        out = self.bn3(out)
        # 下采样
        if self.downsample is not None:
            residual = self.downsample(x)
        # add
        out += residual
        # 激活
        out = self.relu(out)
        # 输出
        return out


class ResNet(nn.Module):
    # 模型构建
    def __init__(self, block, layers, landmarks_num=1000, img_size=224, dropout_factor=1.):
        """
        :param block: 残差结构，层数不同，传入的block也不同，比如说ResNet18/34传入的就是BasicBlock所对应的残差结构，而ResNet50/101/152传入的就是Bottleneck所对应的残差结构
        :param layers: 对应的是一个残差结构的数目，是以列表的形式存储，比如对于ResNet50而言就是[3,4,6,3]
        :param num_classes: 网络输出的类别数
        :param img_size: 图像大小
        :param dropout_factor: 随机失活的概率
        """
        # 定义输入到残差块中特征图的通道，
        # channel定义为64是因为经过maxpool之后channel就会变为64
        self.inplanes = 64
        # 随机失活的概率
        self.dropout_factor = dropout_factor
        super(ResNet, self).__init__()
        # ResNet18、ResNet34、ResNet50、ResNet101等都是先经过由7*7的卷积核、
        # out_channel为64，stride为2的卷积层构成
        # 输入的图片都是3通道的，所以Conv2d的第一个参数为3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # inplace等于true，表示对上层传下来的tensor直接修改，这样能够节省运算内存
        self.relu = nn.ReLU(inplace=True)
        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 连续4个残差模块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 获取特征图大小，整个网络是32倍下采样，所以图像应能够被32整除
        assert img_size % 32 == 0
        pool_kernel = int(img_size / 32)
        # 输出部分：
        # 全局平均池化，输出（n,c,1,1）的特征图
        self.avgpool = nn.AvgPool2d(pool_kernel, stride=1, ceil_mode=True)
        # 随机失活
        self.dropout1 = nn.Dropout(self.dropout_factor)
        self.dropout2 = nn.Dropout(0.8)
        self.dropout3 = nn.Dropout(0.65)

        self.dropout = nn.Dropout(self.dropout_factor)
        # 关键点的输出层
        self.fc_landmarks_1 = nn.Linear(512 * block.expansion, 1024)
        self.fc_landmarks_2 = nn.Linear(1024, landmarks_num)
        # 性别的输出层
        self.fc_gender_1 = nn.Linear(512 * block.expansion, 64)
        self.fc_gender_2 = nn.Linear(64, 2)
        # 年龄的输出层
        self.fc_age_1 = nn.Linear(512 * block.expansion, 64)
        self.fc_age_2 = nn.Linear(64, 1)
        # 人脸姿态输出层
        self.face_ol_1 = nn.Linear(512 * block.expansion, 64)
        self.face_ol_2 = nn.Linear(64, 3)
        for m in self.modules():
            # 卷积层的初始化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # BN层的初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 通过一个1*1卷积进行下采样，在短连接部分
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个残差块中需要shortcut虚线连接，所以要传入downsample和stride
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 第一个残差块的通道数发生变化，要修正通道数
        self.inplanes = planes * block.expansion
        # 添加后续的残差块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # 返回残差模块
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播过程
        # 卷积部分
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 残差模块部分
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 全局池化
        x = self.avgpool(x)
        # 展评为一维向量
        x = x.view(x.size(0), -1)

        # 关键点
        landmarks = self.fc_landmarks_1(x)
        landmarks = self.dropout1(landmarks)
        landmarks = self.fc_landmarks_2(landmarks)
        # 性别
        gender = self.fc_gender_1(x)
        gender = self.dropout2(gender)
        gender = self.fc_gender_2(gender)
        # 年龄
        age = self.fc_age_1(x)
        age = self.dropout3(age)
        age = self.fc_age_2(age)

        ol = self.face_ol_1(x)
        ol = self.face_ol_2(ol)
        # 返回结果
        return landmarks, gender, age, ol


class ResNet2(nn.Module):
    # 模型构建
    def __init__(self, block, layers, landmarks_num=1000, img_size=224, dropout_factor=1.):
        """
        :param block: 残差结构，层数不同，传入的block也不同，比如说ResNet18/34传入的就是BasicBlock所对应的残差结构，而ResNet50/101/152传入的就是Bottleneck所对应的残差结构
        :param layers: 对应的是一个残差结构的数目，是以列表的形式存储，比如对于ResNet50而言就是[3,4,6,3]
        :param num_classes: 网络输出的类别数
        :param img_size: 图像大小
        :param dropout_factor: 随机失活的概率
        """
        # 定义输入到残差块中特征图的通道，
        # channel定义为64是因为经过maxpool之后channel就会变为64
        self.inplanes = 64
        # 随机失活的概率
        self.dropout_factor = dropout_factor
        super(ResNet2, self).__init__()
        # ResNet18、ResNet34、ResNet50、ResNet101等都是先经过由7*7的卷积核、
        # out_channel为64，stride为2的卷积层构成
        # 输入的图片都是3通道的，所以Conv2d的第一个参数为3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # inplace等于true，表示对上层传下来的tensor直接修改，这样能够节省运算内存
        self.relu = nn.ReLU(inplace=True)
        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        # 连续4个残差模块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 获取特征图大小，整个网络是32倍下采样，所以图像应能够被32整除
        assert img_size % 32 == 0
        pool_kernel = int(img_size / 32)
        # 输出部分：
        # 全局平均池化，输出（n,c,1,1）的特征图
        self.avgpool = nn.AvgPool2d(pool_kernel, stride=1, ceil_mode=True)
        # 随机失活
        self.dropout1 = nn.Dropout(self.dropout_factor)
        self.dropout2 = nn.Dropout(0.8)
        self.dropout3 = nn.Dropout(0.65)

        self.dropout = nn.Dropout(self.dropout_factor)
        # 关键点的输出层
        self.fc_landmarks_1 = nn.Linear(512 * block.expansion, 1024)
        self.fc_landmarks_2 = nn.Linear(1024, landmarks_num)
        # 性别的输出层
        self.fc_gender_1 = nn.Linear(512 * block.expansion, 64)
        self.fc_gender_2 = nn.Linear(64, 2)
        # 年龄的输出层
        self.fc_age_1 = nn.Linear(512 * block.expansion, 64)
        self.fc_age_2 = nn.Linear(64, 1)
        for m in self.modules():
            # 卷积层的初始化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # BN层的初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 通过一个1*1卷积进行下采样，在短连接部分
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个残差块中需要shortcut虚线连接，所以要传入downsample和stride
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 第一个残差块的通道数发生变化，要修正通道数
        self.inplanes = planes * block.expansion
        # 添加后续的残差块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # 返回残差模块
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播过程
        # 卷积部分
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 残差模块部分
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 全局池化
        x = self.avgpool(x)
        # 展评为一维向量
        x = x.view(x.size(0), -1)

        # 关键点
        landmarks = self.fc_landmarks_1(x)
        landmarks = self.dropout1(landmarks)
        landmarks = self.fc_landmarks_2(landmarks)
        # 性别
        gender = self.fc_gender_1(x)
        gender = self.dropout2(gender)
        gender = self.fc_gender_2(gender)
        # 年龄
        age = self.fc_age_1(x)
        age = self.dropout3(age)
        age = self.fc_age_2(age)

        # 返回结果
        return landmarks, gender, age


def load_model(model, pretrained_state_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if
                       k in model_dict and model_dict[k].size() == pretrained_state_dict[k].size()}
    model.load_state_dict(pretrained_dict, strict=False)
    if len(pretrained_dict) == 0:
        print("[INFO] No params were loaded ...")
    else:
        for k, v in pretrained_state_dict.items():
            if k in pretrained_dict:
                print("==>> Load {} {}".format(k, v.size()))
            else:
                print("[INFO] Skip {} {}".format(k, v.size()))
    return model


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


def resnet34_2(pretrained=False, **kwargs):
    # 模型初始化
    model = ResNet2(BasicBlock, [3, 4, 6, 3], **kwargs)
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
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        print("Load pretrained model from {}".format(model_urls['resnet152']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        model = load_model(model, pretrained_state_dict)
    return model


class ResNet_teacher(nn.Module):
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
        super(ResNet_teacher, self).__init__()
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
            elif isinstance(m, nn.BatchNorm2d):  # bn层初始化
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


def resnet18_predict_face_ol_function(pretrained=False, **kwargs):
    # 模型初始化
    model = ResNet_teacher(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # 加载预训练模型
        print("Load pretrained model from {}".format(model_urls['resnet18']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        model = load_model(model, pretrained_state_dict)
    return model


if __name__ == "__main__":
    input = torch.randn([32, 3, 256, 256])
    model = resnet50(False, landmarks_num=196, img_size=256)
    landmarks, gender, age, ol = model(input)
    print(landmarks.size(), gender.size(), age.size(), ol.size())
