import torch
import torch.nn as nn

import math


class DCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1,
                 dil=1,
                 groups=1,
                 bias=False):
        super(DCN, self).__init__()
        self.padding = padding
        self.dil = dil
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels//groups, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.groups = groups
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          3 * groups * kernel_size[0] * kernel_size[1],
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dil,
                                          bias=True)

        n = in_channels
        for k in kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

    def offset_mask(self, x):
        tmp = self.conv_offset_mask(x)
        tmp = torch.chunk(tmp, 3, dim=1)
        return torch.cat(tmp[0:2], dim=1), tmp[2]

    def forward(self, x):
        offset, mask = self.offset_mask(x)
        s = mask.size()
        b, c, h, w = s
        mask = mask.view(b, c, h * w)
        mask = torch.softmax(mask, dim=2).view(s)
        output = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.weight.repeat([1,1,3,3]),
                                               bias=self.bias,
                                               padding=self.padding,
                                               dilation=self.dil, mask=mask)
        return output