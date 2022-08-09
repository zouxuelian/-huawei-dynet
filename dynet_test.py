import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
BN_MOMENTUM = 0.1
import time

def conv3x3(in_planes, out_planes, stride=1, g=2):
    """3x3 convolution with padding"""
    return dynamic_convolution(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, g=g)

def conv1x1(in_planes, out_planes):
    """3x3 convolution with padding"""
    return dynamic_convolution(in_planes, out_planes, kernel_size=1, bias=False)


class dynamic_convolution(_ConvNd):
    """仿nn.conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', g=2):
        # 设为2时，64个kernel变为32个kernel，设为4时，64变16
        self.g = g
        # self.one = torch.ones([out_channels, self.g, 1, 1])
        g_out_channels = g * out_channels
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(dynamic_convolution, self).__init__(
            in_channels, g_out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        # self.out_channels = out_channels


    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


    def forward(self, input, coefficient):
        # 测试流程
        weight = torch.mul(coefficient.unsqueeze(2), self.weight.unsqueeze(0))
        # weight [b, out*g, in, k, k] >> [g, b, out, in, k, k] >> [b, out, in, k, k]
        weight = torch.cat(([weight[:, i::self.g, :, :, :].unsqueeze(0) for i in range(self.g)]), 0)
        weight = nn.Parameter(torch.sum(weight, 0).squeeze(0))
        output = self.conv2d_forward(input, weight)
        # 训练流程，之前的流程会导致 weight 的梯度传导不到coefficient中去
        '''output = self.conv2d_forward(input, self.weight)
        output = torch.mul(coefficient, output)
        output = [output[:, i::self.g, :, :].unsqueeze(0) for i in range(self.g)]
        output = torch.cat(output, 0)
        output = torch.sum(output, 0)'''
        return output


class BasicBlock(nn.Module):
    expansion = 1
    '''直接替换resnet中的BasicBlock'''
    def __init__(self, inplanes, planes, stride=1, downsample=None, g=2):
        super(BasicBlock, self).__init__()
        self.g = g
        self.sigmoid = nn.Sigmoid()
        self.planes = planes
        self.coefficient_reduce = nn.Conv2d(in_channels=inplanes, out_channels=inplanes//4, kernel_size=1)
        # 有两层卷积，需要用不同的注意力，所以输出为2倍
        self.coefficient_expand = nn.Conv2d(in_channels=inplanes//4, out_channels=planes*self.g*2, kernel_size=1)

        self.conv1 = conv3x3(inplanes, planes, stride, self.g)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, g=self.g)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        coefficient = torch.sigmoid(self.coefficient_expand(self.relu(self.coefficient_reduce(F.adaptive_avg_pool2d(x, 1)))))

        out = self.conv1(x, coefficient[:, :self.planes*self.g, :, :])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, coefficient[:, self.planes*self.g:, :, :])
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    '''直接替换resnet中的Bottleneck'''
    def __init__(self, inplanes, planes, stride=1, downsample=None, g=2):
        super(Bottleneck, self).__init__()
        self.g = g
        self.sigmoid = nn.Sigmoid()
        self.planes = planes
        self.coefficient_reduce = nn.Conv2d(in_channels=inplanes, out_channels=inplanes // 4, kernel_size=1)
        # 有三层卷积，需要用不同的注意力，所以输出为2+倍
        self.coefficient_expand = nn.Conv2d(in_channels=inplanes // 4, out_channels=planes * self.g * 2 + planes*self.expansion*self.g, kernel_size=1)

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(planes, planes, stride=stride, g=g)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        coefficient = torch.sigmoid(self.coefficient_expand(self.relu(self.coefficient_reduce(F.adaptive_avg_pool2d(x, 1)))))
        out = self.conv1(x, coefficient[:, :self.planes*self.g, :, :])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, coefficient[:, self.planes*self.g:self.planes*self.g*2, :, :])
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out, coefficient[:, self.planes*self.g*2:, :, :])
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

