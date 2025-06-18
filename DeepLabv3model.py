import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义 ResNet 的基本块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 定义 ResNet 骨干网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat


# 定义 ASPP 模块
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                             nn.BatchNorm2d(out_channels),
                                             nn.ReLU())

        self.conv5 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x)))
        x3 = self.relu3(self.bn3(self.conv3(x)))
        x4 = self.relu4(self.bn4(self.conv4(x)))
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu5(self.bn5(self.conv5(x)))
        return x


# 定义 DeepLabv3+ 模型用于细胞核分割
class DeepLabV3PlusForNucleiSegmentation(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3PlusForNucleiSegmentation, self).__init__()
        # 构建 ResNet101 骨干网络
        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3])
        # 构建 ASPP 模块
        self.aspp = ASPP(2048)
        # 低级别特征处理
        self.low_level_conv = nn.Conv2d(256, 48, 1, bias=False)
        self.bn_low_level = nn.BatchNorm2d(48)
        self.relu_low_level = nn.ReLU()
        # 解码器
        self.decoder_conv1 = nn.Conv2d(304, 256, 3, padding=1, bias=False)
        self.bn_decoder1 = nn.BatchNorm2d(256)
        self.relu_decoder1 = nn.ReLU()
        self.decoder_conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn_decoder2 = nn.BatchNorm2d(256)
        self.relu_decoder2 = nn.ReLU()
        self.final_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.size()[2:]  # 记录输入图像的尺寸
        # 骨干网络前向传播
        x, low_level_feat = self.backbone(x)
        # ASPP 模块
        x = self.aspp(x)
        # 上采样到低级别特征的尺寸
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        # 低级别特征处理
        low_level_feat = self.relu_low_level(self.bn_low_level(self.low_level_conv(low_level_feat)))
        # 拼接特征
        x = torch.cat((x, low_level_feat), dim=1)
        # 解码器
        x = self.relu_decoder1(self.bn_decoder1(self.decoder_conv1(x)))
        x = self.relu_decoder2(self.bn_decoder2(self.decoder_conv2(x)))
        # 最终卷积
        x = self.final_conv(x)
        # 上采样到输入图像的尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    # 初始化 DeepLabv3+ 模型，假设分割类别为 1（细胞核）
    model = DeepLabV3PlusForNucleiSegmentation(num_classes=1)
    # 输出模型的具体结构
    print("模型的具体结构:")
    print(model)

    # 输出模型中所有的参数名与参数大小
    print("\n模型中所有的参数名与参数大小:")
    for name, param in model.named_parameters():
        print(f"参数名: {name}, 参数大小: {param.size()}")

    # 模拟输入数据
    input_tensor = torch.randn(2, 3, 256, 256)
    # 前向传播
    output = model(input_tensor)
    print("\n模型输出的形状:")
    print(output.shape)