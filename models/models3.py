
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Dense Block --- #
class DB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        super(DB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        # out = out + x
        return out


class SPP(nn.Module):
    def __init__(self, in_channels):
        super(SPP, self).__init__()
        self.conv1 = nn.Conv2d(5*in_channels, in_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(5*in_channels, in_channels, kernel_size=1, padding=0)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def ups(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        # step = np.sqrt(np.sqrt(np.sqrt(size)))
        # size1 = np.round([2, 2]*step)
        # size2 = np.round([2, 2]*step*step)
        # size3 = np.round([2, 2] * step * step * step)
        p2  = self.ups(self.pool(x, 2), size)
        p4  = self.ups(self.pool(x, 4), size)
        p8  = self.ups(self.pool(x, 8), size)
        p16 = self.ups(self.pool(x, 16), size)
        s   = torch.cat([x, p2, p4, p8, p16], dim=1)
        out = self.conv1(s)
        return out


class rdp_nuc(nn.Module):
    def __init__(self, in_channels=1, depth_rate=16, kernel_size=3, num_dense_layer=4, growth_rate=16):
        super(rdp_nuc, self).__init__()
        self.conv_in1  = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=1)
        self.conv_in2 = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=1)
        self.conv_out1 = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=1)
        self.conv_out2 = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=1)

        self.db1  = DB(depth_rate, num_dense_layer, growth_rate)
        # self.spp11 = SPP(in_channels=depth_rate)
        # self.spp12 = SPP(in_channels=depth_rate)
        self.db2  = DB(depth_rate, num_dense_layer, growth_rate)
        # self.spp21 = SPP(in_channels=depth_rate)
        # self.spp22 = SPP(in_channels=depth_rate)
        self.db3  = DB(depth_rate, num_dense_layer, growth_rate)
        # self.spp31 = SPP(in_channels=depth_rate)
        # self.spp32 = SPP(in_channels=depth_rate)
        self.db4  = DB(depth_rate, num_dense_layer, growth_rate)
        # self.spp41 = SPP(in_channels=depth_rate)
        # self.spp42 = SPP(in_channels=depth_rate)
        self.db5  = DB(depth_rate, num_dense_layer, growth_rate)
        # self.spp51 = SPP(in_channels=depth_rate)
        # self.spp52 = SPP(in_channels=depth_rate)
        self.db6 = DB(depth_rate, num_dense_layer, growth_rate)

        self.spp1 = SPP(in_channels=depth_rate)
        self.spp2 = SPP(in_channels=depth_rate)
        self.spp3 = SPP(in_channels=depth_rate)
        self.spp4 = SPP(in_channels=depth_rate)
        self.spp5 = SPP(in_channels=depth_rate)
        self.spp6 = SPP(in_channels=depth_rate)

    def forward(self, x):
        s0 = self.conv_in1(x)

        s1 = self.db1(s0)
        # p1 = self.spp1(s1)

        b  = self.conv_out1(s1)
        out = x - b

        # s1 = self.spp1(self.db1(s0))
        # s2 = self.spp2(self.db2(s1))
        # s3 = self.spp3(self.db3(s2))
        #
        # LFNU = self.conv_out1(s3)
        #
        # x = x - LFNU
        #
        # s3 = self.conv_in2(x)
        # d3 = self.db4(s3)
        # s4 = torch.sigmoid(self.spp4(d3)) * d3
        #
        # d4 = self.db5(s4)
        # s5 = torch.sigmoid(self.spp5(d4)) * d4
        #
        # d5 = self.db6(s5)
        # s6 = torch.sigmoid(self.spp6(d5)) * d5
        #
        # HFNU = self.conv_out2(s6)

        return out
