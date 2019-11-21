import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math

class Detach_max(nn.Module):
    def __init__(self):
        super(Detach_max,self).__init__()

    def forward(self, input):
        max_value = torch.max(torch.max(torch.max(torch.abs(input),\
                            0,True)[0], 2, True)[0], -1, True)[0]
        out = input / max_value * torch.detach(max_value)

        return out
class MyBatchNormFunction(torch.autograd.Function):

    # 必须是staticmethod
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    # ctx在这里类似self，ctx的属性可以在backward中调用。
    def forward(self,x,mean,var,eps,alpha,beta):
        self.save_for_backward(x,mean,var,torch.tensor(eps),alpha,beta)
        return (x - mean) / torch.sqrt(var + eps) * alpha + beta
    @staticmethod
    def backward(self, grad_output):

        x, mean, var, eps, alpha, beta = self.saved_variables
        batch_size = x.shape[0]
        dL_dXhat = grad_output * alpha
        dL_dSigama = torch.sum(dL_dXhat * (x - mean) * (-0.5) * \
                               torch.pow(var + eps.item(),-1.5), dim=0,keepdim=True)

        I = dL_dXhat / torch.sqrt(var + eps.item()) + dL_dSigama * 2 * (x - mean) / batch_size

        dL_dmean = - torch.sum(I,dim=0,keepdim=True)

        dL_dX = I + 1/batch_size*dL_dmean

        x_hat = (x - mean)/torch.sqrt(var)
        dL_dAlpha = torch.sum(x_hat * grad_output,dim=0,keepdim=True)
        dL_dBeta = torch.sum(grad_output,dim=0,keepdim=True)
        return dL_dX,None,None,None,dL_dAlpha,dL_dBeta

class MyBatchNorm(nn.Module):
    def __init__(self,num_features,affine=False):
        super(MyBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = 1e-05
        self.momente = 0.9
        self.running_mean = nn.Parameter(torch.zeros([1, self.num_features, 1, 1]))
        self.running_var = nn.Parameter(torch.ones([1, self.num_features, 1, 1]))
        self.affine = affine

        if self.affine:
            self.alpha = nn.Parameter(torch.ones([1, self.num_features, 1, 1]))
            self.beta = nn.Parameter(torch.zeros([1, self.num_features, 1, 1]))
        else:
            self.alpha = nn.Parameter(torch.ones([1, self.num_features, 1, 1]),\
                                      requires_grad=False)
            self.beta = nn.Parameter(torch.zeros([1, self.num_features, 1, 1]),\
                                     requires_grad=False)
        self.training = True
    def forward(self, x):
        if self.training:
            mean = torch.mean(torch.mean(torch.mean(x, \
                                                    0, True), 2, True), -1, True)
            # print(mean[0,:,0,0])
            var = torch.mean(torch.mean(torch.mean((x -\
                mean) ** 2, 0, True), 2, True), -1, True) + self.eps
            self.running_mean = nn.Parameter(self.running_mean * self.momente + mean * \
                                             (1 - self.momente))
            self.running_var = nn.Parameter(self.running_var * self.momente + var * \
                                            (1 - self.momente))

            output = MyBatchNormFunction(x,mean,var,self.eps,self.alpha,self.beta)
            return output
        else:
            x_hat = (x - self.running_mean)/ torch.sqrt(self.running_var + self.eps)
            return x_hat * self.alpha + self.beta



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
