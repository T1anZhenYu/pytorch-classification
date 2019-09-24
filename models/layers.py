import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv2d_new(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_new, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)

        self.moving_var = nn.Parameter(torch.ones([out_channels,1,1,1]), requires_grad=False)
        self.momente = 0.1
        self.alpha = torch.ones([out_channels,1,1,1]).cuda()+\
                     torch.abs(nn.Parameter(torch.ones([out_channels,1,1,1]), requires_grad=True).cuda())
    def forward(self, x):  # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean

        out1 = F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        eps = 1e-5
        shape_2d = (1,out1.shape[1],1, 1)
        mu = torch.mean(out1, dim=(0, 2, 3)).view(shape_2d)
        var = torch.transpose(torch.mean(
            (out1 - mu) ** 2, dim=(0, 2, 3)).view(shape_2d), 0, 1) # biased
        self.moving_var = nn.Parameter(self.momente*self.moving_var +
                                       (1-self.momente)*var,requires_grad=False)
        return F.conv2d(x, weight /(self.alpha*torch.sqrt(self.moving_var + eps)), \
                        self.bias, self.stride,self.padding, self.dilation, self.groups)


def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)
