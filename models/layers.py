import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math

class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps
        self.moving_mean = torch.zeros(num_features)
        self.moving_var = torch.ones(num_features)
        self.momente = 0.9
    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        if C < G:
            G = C

        assert C % G == 0

        x = x.view(N,G,-1)
        temp_mean = x.mean(-1, keepdim=True)
        temp_var = x.var(-1, keepdim=True)
        mean = temp_mean*(1-self.momente) + self.momente*self.moving_mean

        var = temp_var*(1-self.momente) + self.moving_var*self.momente
        if self.training:
            self.moving_mean = mean
            self.moving_var = var
        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias


class MYBatchNorm(nn.Module):
    '''custom implement batch normalization with autograd by Antinomy
    '''

    def __init__(self, num_features):
        super(MYBatchNorm, self).__init__()
        # auxiliary parameters
        self.num_features = num_features
        self.eps = 1e-5
        self.momentum = 0.1
        # hyper paramaters
        self.gamma = nn.Parameter(torch.Tensor(self.num_features), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(self.num_features), requires_grad=True)
        # moving_averge
        self.moving_mean = torch.zeros(self.num_features)
        self.moving_var = torch.ones(self.num_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.moving_var)
        nn.init.zeros_(self.moving_mean)

    def forward(self, X):
        assert len(X.shape) in (2, 4)
        if X.device.type != 'cpu':
            self.moving_mean = self.moving_mean.cuda()
            self.moving_var = self.moving_var.cuda()
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta,
                                         self.moving_mean, self.moving_var,
                                         self.training, self.eps, self.momentum)
        return Y


def batch_norm(X, gamma, beta, moving_mean, moving_var, is_training=True, eps=1e-5, momentum=0.9,):

    if len(X.shape) == 2:
        mu = torch.mean(X, dim=0)
        var = torch.mean((X - mu) ** 2, dim=0)
        if is_training:
            X_hat = (X - mu) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mu
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        else:
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        out = gamma * X_hat + beta

    elif len(X.shape) == 4:
        shape_2d = (1, X.shape[1], 1, 1)
        mu = torch.mean(X, dim=(0, 2, 3)).view(shape_2d)
        var = torch.mean(
            (X - mu) ** 2, dim=(0, 2, 3)).view(shape_2d) # biased

        if is_training:
            X_hat = (X - mu) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean.view(shape_2d) + (1.0 - momentum) * mu
            moving_var = momentum * moving_var.view(shape_2d) + (1.0 - momentum) * var
        else:
            X_hat = (X - moving_mean.view(shape_2d)) / torch.sqrt(moving_var.view(shape_2d) + eps)

        out = gamma.view(shape_2d) * X_hat + beta.view(shape_2d)

    return out, moving_mean, moving_var


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
        self.gamma = nn.Parameter(torch.ones([1,out_channels,1,1]),requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1,out_channels,1,1]),requires_grad=True)
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
        shape_2d = (1,out1.shape[1],1, 1)
        # estimate_mean = (torch.sqrt(torch.tensor(math.pi/2)).cuda()/\
        #                 (estimate_max/real_max+self.eps)).view([1,x.shape[1],1,1]).cuda()

        # weight = weight - weight_mean


        mu = torch.mean(out1, dim=(0, 2, 3)).view(shape_2d)
        var = torch.mean(
            (out1 - mu) ** 2, dim=(0, 2, 3)).view(shape_2d)
        return (out1-mu)/torch.sqrt(var+self.eps)
        # weight = weight /(torch.sqrt(var+self.eps))
        # weight = (1-self.momente) * (weight/(torch.sqrt(var+self.eps))) + (self.momente)*self.weight
        # real_out = F.conv2d(x, weight, \
        #                 self.bias, self.stride,self.padding, self.dilation, self.groups)
        # return self.gamma*(real_out-mu/(torch.transpose(torch.sqrt(var+self.eps),0,1)))

def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=16)
