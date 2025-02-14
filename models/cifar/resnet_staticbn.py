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
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from .. import layers as L
__all__ = ['resnet_staticbn']

writer = SummaryWriter('tensorboard_data')
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = L.MyStaticBatchNorm(planes,True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = L.MyStaticBatchNorm(planes,False)
        self.downsample = downsample
        self.stride = stride
        self.iter = 1
        # self.estimate_mean = nn.Parameter(torch.zeros([planes]))
        # self.estimate_var = nn.Parameter(torch.ones([planes]))
    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out0 = self.bn1(out,self.conv1.weight,x)
        out1 = self.relu(out0)
        out2 = self.conv2(out1)
        out3 = self.bn2(out2,self.conv2.weight,out1)
        if self.iter %200 == 1 and self.training and self.conv1.in_channels == 32 and \
                self.conv1.out_channels == 64:
            writer.add_histogram(tag='input',values=out1,global_step=self.iter)
            writer.add_histogram(tag='conv2_weight', values=self.conv2.weight,\
                                 global_step=self.iter)
            writer.add_histogram(tag='output',values=out2,global_step=self.iter)
            writer.add_histogram(tag='afterBN',values=out3,global_step=self.iter)
        # c_in = self.conv1.out_channels
        # weight_mean = torch.mean(self.conv2.weight,(1,2,3))
        # weight_var = torch.var(self.conv2.weight,(1,2,3))
        # dic = {}
        # real_max = torch.max(torch.max(torch.max(out1,dim=0)[0],dim=-1)[0],dim=-1)[0]
        # estimate_max = 0.83*math.log(out1.shape[0]*out1.shape[1]*out1.shape[2]*out1.shape[3])
        # alpha = real_max / estimate_max
        # estimate_mean = c_in * math.sqrt(math.pi/2) * weight_mean
        # estimate_var = alpha**2 * c_in**2 * math.pi /2 * weight_var
        # if self.iter == 1:
        #     self.estimate_mean = nn.Parameter(estimate_mean,requires_grad=False)
        #     self.estimate_var = nn.Parameter(estimate_var,requires_grad=False)
        # else:
        #     self.estimate_mean = nn.Parameter(estimate_mean*0.1 + self.estimate_mean*0.9,requires_grad=False)
        #
        #     self.estimate_var = nn.Parameter(estimate_var*0.1 + self.estimate_var*0.9,requires_grad=False)
        #
        # if self.iter[0] % 200 == 0 and self.conv1.in_channels != self.conv1.out_channels:
        #     dic['alpha_C'+str(alpha)] = alpha.detach().cpu().numpy()
        #     dic['running_estimate_mean_C'+str(self.conv1.out_channels)] = \
        #         self.estimate_mean.detach().cpu().numpy()
        #     dic['running_estimate_var_C'+str(self.conv1.out_channels)] = \
        #         self.estimate_var.detach().cpu().numpy()
        #     dic['temp_estimate_mean_C'+str(self.conv1.out_channels)] = \
        #         estimate_mean.detach().cpu().numpy()
        #     dic['temp_estimate_var_C'+str(self.conv1.out_channels)] = \
        #         estimate_var.detach().cpu().numpy()
        #     dic['input_C'+str(self.conv1.out_channels)] = out1.detach().cpu().numpy()
        #     dic['beforeBN_C'+str(self.conv1.out_channels)] = out2.detach().cpu().numpy()
        #     dic['afterBN_C'+str(self.conv1.out_channels)] = out3.detach().cpu().numpy()
        #     dic['weight_C'+str(self.conv1.out_channels)] = self.conv2.weight.detach().cpu().numpy()
        #     dic['conv1weight_C'+str(self.conv1.out_channels)] = self.conv1.weight.detach().cpu().numpy()
        #     dic['iter_C'+str(self.conv1.out_channels)] = self.iter.detach().cpu().numpy()
        #     np.savez(str(time.time()) + '_C'+str(self.conv1.out_channels)+'.npz', **dic)
        # if self.training:
        #     self.iter = nn.Parameter(self.iter + 1, requires_grad= False)

        if self.downsample is not None:
            residual = self.downsample(x)
        self.iter += 1
        out3 += residual
        out4 = self.relu(out3)

        return out4


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = L.MyStaticBatchNorm(planes,True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = L.MyStaticBatchNorm(planes,False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = L.MyStaticBatchNorm(planes * 4,False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out1 = self.bn1(out,self.conv1.weight,x)
        out2 = self.relu(out1)

        out3 = self.conv2(out2)
        out4 = self.bn2(out3,self.conv2.weight,out2)
        out4 = self.relu(out4)

        out5 = self.conv3(out4)
        out = self.bn3(out5,self.conv3.weight,out4)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetStaticBN(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNetStaticBN, self).__init__()
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16,affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine=False),
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
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_staticbn(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNetStaticBN(**kwargs)
