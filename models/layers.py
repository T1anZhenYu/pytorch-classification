import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps
        if num_groups > num_features:
            self.moving_mean = torch.zeros([1, num_features, 1]).cuda().detach()
            self.moving_var = torch.ones([1, num_features, 1]).cuda().detach()
        else:
            self.moving_mean = torch.zeros([1, num_groups, 1]).cuda().detach()
            self.moving_var = torch.ones([1, num_groups, 1]).cuda().detach()
        self.momente = 0.9

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        if C < G:
            G = C

        assert C % G == 0

        x = x.view(N, G, -1)
        temp_mean = x.mean(-1, keepdim=True)
        temp_var = x.var(-1, keepdim=True)
        mean = temp_mean * (self.momente) + (1 - self.momente) * self.moving_mean

        var = temp_var * (self.momente) + self.moving_var * (1 - self.momente)
        if self.training:
            self.moving_mean = (temp_mean * (1 - self.momente) +
                                self.momente * self.moving_mean).detach()
            self.moving_var = (temp_var * (1 - self.momente) +
                               self.moving_var * self.momente).detach()
        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
class Detach_max(nn.Module):
    def __init__(self):
        super(Detach_max,self).__init__()

    def forward(self, input):
        print('input shape:')
        print(input.shape)
        max_value = torch.max(torch.max(torch.max(torch.abs(input),\
                            -1,True)[0], -1, True)[0], -1, True)[0]
        out = input / max_value * torch.detach(max_value)

        return out
# class MyBatchNorm(nn.Module):
#     def __init__(self, num_features):
#         super(MyBatchNorm, self).__init__()
#         self.num_features = num_features
#         self.eps = 1e-5
#         self.momente = 0.9
#         self.running_mean = nn.Parameter(torch.zeros([1, self.num_features, 1, 1]))
#         self.running_var = nn.Parameter(torch.ones([1, self.num_features, 1, 1]))
#
#
#     def forward(self, x):
#         input_shape = x.shape
#         if len(input_shape) == 4:
#             mean = torch.mean(x, dim=-1, )

class MyStaticBatchNorm(nn.Module):
    def __init__(self, num_features, residual=True):
        super(MyStaticBatchNorm, self).__init__()
        self.residual = residual
        self.num_features = num_features
        self.eps = 1e-5
        self.momente = 0.9
        self.running_mean = nn.Parameter(torch.zeros([1, self.num_features, 1, 1]))

        self.running_var = nn.Parameter(torch.ones([1, self.num_features, 1, 1]))
        self.running_alpha = nn.Parameter(torch.ones([1]), requires_grad=False)

    def forward(self, x, last_layer_weight, last_layer_input):
        c_in = last_layer_input.shape[1]
        weight_mean = torch.mean(last_layer_weight, (1, 2, 3))
        weight_var = torch.var(last_layer_weight, (1, 2, 3))
        real_max = torch.mean(torch.max(torch.max(torch.max(last_layer_input, dim=-1)[0] \
                                                  , dim=-1)[0], dim=-1)[0])
        estimate_max = 0.83 * math.log(last_layer_input.shape[0] * \
                                       last_layer_input.shape[1] * \
                                       last_layer_input.shape[2] * \
                                       last_layer_input.shape[3])
        # alpha = self.alpha * self.momente + (1-self.momente) * real_max / estimate_max

        # real_mean = torch.mean(x,(0,2,3)).view([1,self.num_features,1,1])
        if self.residual:
            estimate_mean = (c_in * math.sqrt(math.pi / 2) * weight_mean * 2) \
                .view([1, self.num_features, 1, 1])
            estimate_var = (c_in ** 2 * math.pi / 2 * weight_var * 4) \
                .view([1, self.num_features, 1, 1])
        else:
            estimate_mean = (c_in * math.sqrt(math.pi / 2) * weight_mean) \
                .view([1, self.num_features, 1, 1])
            estimate_var = (c_in ** 2 * math.pi / 2 * weight_var) \
                .view([1, self.num_features, 1, 1])
        if self.training:
            alpha = real_max / estimate_max
            self.running_alpha = nn.Parameter(self.momente * self.running_alpha + \
                                              (1 - self.momente) * alpha, requires_grad=False)
            self.running_mean = nn.Parameter(self.momente * self.running_mean + \
                                             (1 - self.momente) * estimate_mean)
            self.running_var = nn.Parameter(self.running_var * self.momente + \
                                            (1 - self.momente) * alpha ** 2 * estimate_var)
            return (x - estimate_mean) / torch.sqrt(self.running_alpha ** 2 * estimate_var \
                                                    + self.eps)
        else:
            # temp_estimate_mean = self.running_mean*self.momente + \
            #                      (1-self.momente)*estimate_mean
            # temp_estimate_var = self.running_var * self.momente + \
            #                     (1-self.momente) * estimate_var
            return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)


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

        self.momente = 0.01
        self.gamma = nn.Parameter(torch.ones([1, out_channels, 1, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, out_channels, 1, 1]), requires_grad=True)
        self.eps = 1e-5

    def forward(self, x):  # return super(Conv2d, self).forward(x)
        weight = self.weight
        # weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
        #                                             keepdim=True).mean(dim=3, keepdim=True)

        # total_num = x.shape[0] * x.shape[1] * x.shape[2] *x.shape[3]
        # estimate_max = (0.82 * torch.log(torch.tensor(total_num,dtype= torch.float))).cuda()
        # real_max = torch.max(x).cuda()

        out1 = F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        shape_2d = (1, out1.shape[1], 1, 1)
        # estimate_mean = (torch.sqrt(torch.tensor(math.pi/2)).cuda()/\
        #                 (estimate_max/real_max+self.eps)).view([1,x.shape[1],1,1]).cuda()

        # weight = weight - weight_mean

        mu = torch.mean(out1, dim=(0, 2, 3)).view(shape_2d)
        var = torch.mean(
            (out1 - mu) ** 2, dim=(0, 2, 3)).view(shape_2d)
        return (out1 - mu) / torch.sqrt(var + self.eps)
        # weight = weight /(torch.sqrt(var+self.eps))
        # weight = (1-self.momente) * (weight/(torch.sqrt(var+self.eps))) + (self.momente)*self.weight
        # real_out = F.conv2d(x, weight, \
        #                 self.bias, self.stride,self.padding, self.dilation, self.groups)
        # return self.gamma*(real_out-mu/(torch.transpose(torch.sqrt(var+self.eps),0,1)))


def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=16)
