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


def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)



# class TLU(nn.Module):
#     def __init__(self, num_features):
#         """
#         max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau
#         """
#         super(TLU, self).__init__()
#
#         self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), \
#                                           requires_grad=True)
#
#     def reset_parameters(self):
#         nn.init.zeros_(self.tau)
#
#     def forward(self, x):
#         return F.relu(x - self.tau) + self.tau
#
#     def extra_repr(self):
#         inplace_str = 'inplace=True' if self.inplace else ''
#         return inplace_str
#
#
# class FRN(nn.Module):
#     def __init__(self, num_features, init_eps=1e-6):
#         """
#         weight = gamma, bias = beta
#         beta, gamma:
#             Variables of shape [1, 1, 1, C]. if TensorFlow
#             Variables of shape [1, C, 1, 1]. if PyTorch
#         eps: A scalar constant or learnable variable.
#         """
#         super(FRN, self).__init__()
#
#         self.num_features = num_features
#         self.init_eps = init_eps
#
#         self.weight = nn.parameter.Parameter(
#             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
#         self.bias = nn.parameter.Parameter(
#             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
#         self.eps = nn.parameter.Parameter(
#             torch.Tensor(1), requires_grad=False)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.ones_(self.weight)
#         nn.init.zeros_(self.bias)
#         nn.init.zeros_(self.eps)
#
#     def extra_repr(self):
#         return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)
#
#     def forward(self, x):
#         """
#         0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
#         0, 1, 2, 3 -> (B, C, H, W) in PyTorch
#         TensorFlow code
#             nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
#             x = x * tf.rsqrt(nu2 + tf.abs(eps))
#             # This Code include TLU function max(y, tau)
#             return tf.maximum(gamma * x + beta, tau)
#         """
#         # Compute the mean norm of activations per channel.
#         nu2 = (x ** 2).mean(dim=[2, 3], keepdim=True)
#
#         # Perform FRN.
#         x = x * (nu2 + self.init_eps + self.eps.abs())**(-0.5)
#
#         # Scale and Bias
#         x = self.weight * x + self.bias
#         return x

class FilterResponseNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(FilterResponseNormalization, self).__init__()
        self.beta = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
             torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(torch.Tensor([eps]),requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """

        n, c, h, w = x.shape
        assert (self.gamma.shape[1], self.beta.shape[1], self.tau.shape[1]) == (c, c, c)

        # Compute the mean norm of activations per channel
        nu2 = x.pow(2).mean(dim=(2,3), keepdim=True)
        # Perform FRN
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        # Return after applying the Offset-ReLU non-linearity
        return torch.max(self.gamma*x + self.beta, self.tau)