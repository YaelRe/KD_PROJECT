'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import math

import torch
import torch.nn as nn

from layers import NoisedConv2DColored as Conv2d
from layers import NoisedLinear as Linear


def conv3x3(in_planes, out_planes, stride=1, act_dim_a=None, act_dim_b=None, weight_noise=False, act_noise_a=False,
            act_noise_b=False, rank=5, noised_strength=0.25, noisef_strength=0.1):
    " 3x3 convolution with padding "
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, act_dim_a=act_dim_a,
                  act_dim_b=act_dim_b, weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b,
                  rank=rank, noised_strength=noised_strength, noisef_strength=noisef_strength)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_dim_a=None, act_dim_b=None, weight_noise=False,
                 act_noise_a=False, act_noise_b=False, rank=5, noised_strength=0.25, noisef_strength=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, act_dim_a, act_dim_b, weight_noise=weight_noise,
                             act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank,
                             noised_strength=noised_strength, noisef_strength=noisef_strength)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1, act_dim_b, act_dim_b, weight_noise=weight_noise,
                             act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank,
                             noised_strength=noised_strength, noisef_strength=noisef_strength)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_dim_a=None, act_dim_b=None, weight_noise=False,
                 act_noise_a=False, act_noise_b=False, rank=5, noised_strength=0.25, noisef_strength=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False, act_dim_a=act_dim_a, act_dim_b=act_dim_a,
                            weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank,
                            noised_strength=noised_strength, noisef_strength=noisef_strength)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, act_dim_a=act_dim_a,
                            act_dim_b=act_dim_b, weight_noise=weight_noise, act_noise_a=act_noise_a,
                            act_noise_b=act_noise_b, rank=rank,
                            noised_strength=noised_strength, noisef_strength=noisef_strength)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False, act_dim_a=act_dim_b, act_dim_b=act_dim_b,
                            weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank,
                            noised_strength=noised_strength, noisef_strength=noisef_strength)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_dim_a=None, act_dim_b=None, weight_noise=False,
                 act_noise_a=False, act_noise_b=False, rank=5):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride, act_dim_a=act_dim_a, act_dim_b=act_dim_b,
                             weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, 1, act_dim_a=act_dim_b, act_dim_b=act_dim_b, weight_noise=weight_noise,
                             act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act_dim_a=None, act_dim_b=None, weight_noise=False,
                 act_noise_a=False, act_noise_b=False, rank=5):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False, act_dim_a=act_dim_a, act_dim_b=act_dim_a,
                            weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, act_dim_a=act_dim_a,
                            act_dim_b=act_dim_b, weight_noise=weight_noise, act_noise_a=act_noise_a,
                            act_noise_b=act_noise_b, rank=rank)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False, act_dim_a=act_dim_b, act_dim_b=act_dim_b,
                            weight_noise=weight_noise, act_noise_a=act_noise_a, act_noise_b=act_noise_b, rank=rank)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):
    def __init__(self, block, layers, width=1, num_classes=10, input_size=32, weight_noise=False, act_noise_a=False,
                 act_noise_b=False, rank=5, noised_strength=0.25, noisef_strength=0.1):
        super(ResNet_Cifar, self).__init__()

        self.weight_noise = weight_noise
        self.act_noise_a = act_noise_a
        self.act_noise_b = act_noise_b
        self.rank = rank
        self.noised_strength = noised_strength
        self.noisef_strength = noisef_strength

        inplanes = int(16 * width)
        self.inplanes = inplanes
        self.conv1 = Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False, act_dim_a=input_size,
                            act_dim_b=input_size, weight_noise=self.weight_noise, act_noise_a=self.act_noise_a,
                            act_noise_b=self.act_noise_b, rank=self.rank, noised_strength=self.noised_strength,
                            noisef_strength=self.noisef_strength)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0], input_size=input_size)
        self.layer2 = self._make_layer(block, 2 * inplanes, layers[1], stride=2, input_size=input_size)
        self.layer3 = self._make_layer(block, 4 * inplanes, layers[2], stride=2, input_size=input_size // 2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = Linear(4 * inplanes * block.expansion, num_classes, weight_noise=self.weight_noise,
                         noise_strength=self.noised_strength)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, input_size=32):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                       act_dim_a=input_size, act_dim_b=input_size // stride, weight_noise=self.weight_noise,
                       act_noise_a=self.act_noise_a, act_noise_b=self.act_noise_b, rank=self.rank,
                       noised_strength=self.noised_strength, noisef_strength=self.noisef_strength),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, act_dim_a=input_size, act_dim_b=input_size // stride,
                  weight_noise=self.weight_noise, act_noise_a=self.act_noise_a, act_noise_b=self.act_noise_b,
                  rank=self.rank, noised_strength=self.noised_strength, noisef_strength=self.noisef_strength))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, act_dim_a=input_size // stride, act_dim_b=input_size // stride,
                                weight_noise=self.weight_noise, act_noise_a=self.act_noise_a,
                                act_noise_b=self.act_noise_b, rank=self.rank, noised_strength=self.noised_strength,
                                noisef_strength=self.noisef_strength))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        # print(x.grad_fn)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print(x.grad_fn)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(x.grad_fn)

        return x


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, weight_noise=False, act_noise_a=False, act_noise_b=False, rank=5):
        super(PreAct_ResNet_Cifar, self).__init__()

        self.weight_noise = weight_noise
        self.act_noise_a = act_noise_a
        self.act_noise_b = act_noise_b
        self.rank = rank

        self.inplanes = 16
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = Linear(64 * block.expansion, num_classes, noise_strength=self.linear_noise_strength)

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                       weight_noise=self.weight_noise, act_noise_a=self.act_noise_a, act_noise_b=self.act_noise_b,
                       rank=self.rank)
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, weight_noise=self.weight_noise,
                  act_noise_a=self.act_noise_a, act_noise_b=self.act_noise_b, rank=self.rank))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, weight_noise=self.weight_noise, act_noise_a=self.act_noise_a,
                                act_noise_b=self.act_noise_b))
        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO: normalization
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet20_cifar()
    y = net(torch.randn(1, 3, 32, 32))
    print(net)
    print(y.size())
