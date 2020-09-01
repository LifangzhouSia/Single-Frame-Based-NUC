import torch
import torch.nn as nn
import numpy as np
from torch import autograd
import torch.nn.functional as F
from models.attention_block import CBAM
from models.nonlocal_block import NonLocalBlock


class doubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_merge(nn.Module):
    def __init__(self, tc, mc):
        super().__init__()
        self.att = CBAM(total_channels=tc, merged_channels=mc)
        self.conv = doubleConv(ch_in=tc, ch_out=mc)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=tc, out_channels=mc, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(inplace=True))

    def forward(self, x, y):
        size = x.size()[2:]
        yr = F.interpolate(y, size, mode='bilinear', align_corners=True)
        xy = torch.cat([x, yr], dim=1)
        z  = self.conv(xy)
        csa = self.att(xy, 64)
        return csa*z


def adaptive_pool_list(height, width, L, bottom_size):
    bh, bw = bottom_size
    sh = pow(height/bh, 1/L)
    sw = pow(width/bw, 1/L)
    hlist = [int(round(bh*pow(sh, k))) for k in range(L)]
    wlist = [int(round(bw*pow(sw, k))) for k in range(L)]
    hlist[L-1] = height
    wlist[L-1] = width
    hlist.reverse()
    wlist.reverse()
    return hlist, wlist

def regular_pool_list(height, width, L):
    hlist = [int(height//pow(2, k)) for k in range(L)]
    wlist = [int(width//pow(2, k)) for k in range(L)]
    return hlist, wlist


class AUNet(nn.Module):
    def __init__(self, height, width, L, bottom_size=[4, 4]):
        super().__init__()

        nb_filter = [64 for _ in range(L)]
        # hlist, wlist = adaptive_pool_list(height, width, L, bottom_size)
        hlist, wlist = regular_pool_list(height, width, L)

        self.downsample_module = nn.ModuleDict()
        self.attmerge_module = nn.ModuleDict()
        self.merge_module = nn.ModuleDict()

        self.conv_in = doubleConv(1, nb_filter[0])
        self.conv_out = nn.Conv2d(nb_filter[0], 1, kernel_size=3, stride=1, padding=1)

        for k in range(1, L):
            self.downsample_module.update({'{}'.format(k): nn.Sequential(
                nn.AdaptiveAvgPool2d([hlist[k], wlist[k]]),
                doubleConv(nb_filter[k-1], nb_filter[k]))})

        self.nlb_btm = NonLocalBlock(nb_filter[L-1])
        self.att_btm = CBAM(total_channels=nb_filter[L-1], merged_channels=nb_filter[L-1])

        for k in range(L-2, -1, -1):
            self.attmerge_module.update({'{}'.format(k): Attention_merge(
                tc=nb_filter[k+1] + nb_filter[k],
                mc=nb_filter[k])})

    def forward(self, x):

        L = 5

        x_idx = [0 for _ in range(L)]
        y_idx = [0 for _ in range(L)]
        x_idx[0] = self.conv_in(x)

        for k in range(1, L):
            x_idx[k] = self.downsample_module['{}'.format(k)](x_idx[k-1])

        y_idx[L-1] = self.nlb_btm(self.att_btm(x_idx[L-1], 64)*x_idx[L-1])

        for k in range(L-2, -1, -1):
            y_idx[k] = self.attmerge_module['{}'.format(k)](x_idx[k], y_idx[k+1])

        y = self.conv_out(y_idx[0])

        # for k in range(L):
        #     print('{}, {}'.format(x_idx[k].size(), y_idx[k].size()))

        out = x - y

        return out
