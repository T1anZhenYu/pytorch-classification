from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch
import torch.nn as nn
import math
from .. import layers as L
import numpy as np
import time

__all__ = ['resnet_origin_gn']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return L.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return L.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = L.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = L.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.iter = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        residual = x
        dic = {}
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        dic['iter'] = self.iter.cpu().numpy()
        dic['input'] = out.detach().cpu().numpy()
        out = self.conv2(out)
        dic['weight'] = self.conv2.weight.detach().cpu().numpy()
        dic['beforeBN'] = out.detach().cpu().numpy()
        out = self.bn2(out)
        dic['afterBN'] = out.detach().cpu().numpy()
        if self.iter % 50 == 0 and self.conv2.in_channels == 16:
            np.savez(str(time.time()) + ".npz", **dic)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        self.iter = nn.Parameter(self.iter + 1, requires_grad=False)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)

        self.bn1 = L.BatchNorm2d(planes)
        self.conv2 = L.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = L.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes * 4, )
        self.bn3 = L.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        dic = {}

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
        np.savez(str(time.time()) + ".npz", **dic)
        return out


class ResNetOriginGN(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNetOriginGN, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = L.Conv2d(3, 16, kernel_size=3, padding=1,
                              bias=False)
        self.bn1 = L.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, L.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                L.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                L.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        dic = {}

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_origin_gn(**kwargs):
    """
    Constructs a ResNetOriginGN model.
    """
    return ResNetOriginGN(**kwargs)
