
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vertical_nl_block import VerticalNLBlock


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
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        return out


class basicConv(nn.Module):
    def __init__(self, in_channels):
        super(basicConv, self).__init__()
        self.bc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.bc(x)
        out = x + out
        return out


class basicAttention(nn.Module):
    def __init__(self, in_channels, att_channels):
        super(basicAttention, self).__init__()
        self.ba = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, att_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.ba(x)


class rdp_nuc(nn.Module):
    def __init__(self, in_channels=1, depth_rate=32, kernel_size=3, num_dense_layer=4, growth_rate=16):
        super(rdp_nuc, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=1)
        self.db_out = DB(depth_rate, num_dense_layer, growth_rate)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=1)

        self.db1 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db2 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db3 = DB(depth_rate, num_dense_layer, growth_rate)
        self.gff_1x1 = nn.Conv2d(depth_rate*3, depth_rate, kernel_size=1, padding=0)

        self.att0 = basicAttention(depth_rate, depth_rate)
        self.att2 = basicAttention(depth_rate, depth_rate)
        self.att4 = basicAttention(depth_rate, depth_rate)
        self.att8 = basicAttention(depth_rate, depth_rate)
        self.att16 = basicAttention(depth_rate, depth_rate)

        self.bc0 = basicConv(depth_rate)
        self.bc2 = basicConv(depth_rate)
        self.bc4 = basicConv(depth_rate)
        self.bc8 = basicConv(depth_rate)
        self.bc16 = basicConv(depth_rate)

        self.rb = nn.Conv2d(depth_rate*4, depth_rate, kernel_size=1, padding=0)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def ups(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x, h0, h2, h4, h8, b0, b2, b4, b8, is_init=False):

        size = x.size()[2:]
        s1 = self.conv_in(x)

        # --- Dense block --- #
        d1 = self.db1(s1)
        d2 = self.db2(d1)
        d3 = self.db3(d2)
        dc = torch.cat([d1, d2, d3], dim=1)
        s2 = self.gff_1x1(dc)

        # --- High-Frequency Noise Reduction ---#
        if is_init:
            b0 = self.att0(s2)
            h0 = self.bc0(s2)
        else:
            a0 = self.att0(s2)
            g0 = self.bc0(s2)
            b0 = b0 / (0.001*b0 + 1)
            h0 = a0 / (a0 + b0) * g0 + b0 / (a0 + b0) * h0
        s2 = s2 + h0

        # --- Pyramid block --- #
        p2 = self.pool(s2, 2)
        p4 = self.pool(s2, 4)
        p8 = self.pool(s2, 8)
        p16 = self.pool(s2, 16)

        # --- Pyramid attention block --- #
        if is_init:
            b2 = self.att2(p2)
            b4 = self.att4(p4)
            b8 = self.att8(p8)
            b16 = self.att16(p16)
        else:
            a2 = self.att2(p2)
            a4 = self.att4(p4)
            a8 = self.att8(p8)
            a16 = self.att16(p16)

        # --- Pyramid convolution block --- #

        if is_init:
            h2 = b2*self.bc2(p2)
            h4 = b4 * self.bc4(p4)
            h8 = b8 * self.bc8(p8)
            h16 = b16 * self.bc16(p16)

        else:
            g2 = self.bc2(p2)
            b2 = b2 / (0.001 * b2 + 1)
            h2 = a2 / (a2 + b2) * g2 + b2 / (a2 + b2) * h2

            g4 = self.bc4(p4)
            b4 = b4 / (0.001 * b4 + 1)
            h4 = a4 / (a4 + b4) * g4 + b4 / (a4 + b4) * h4

            g8 = self.bc8(p8)
            b8 = b8 / (0.001 * b8 + 1)
            h8 = a8 / (a8 + b8) * g8 + b8 / (a8 + b8) * h8

            g16 = self.bc16(p16)
            b16 = b16 / (0.001 * b16 + 1)
            h16 = a16 / (a16 + b16) * g16 + b16 / (a16 + b16) * h16

        # --- Merge the multi-scale features --- #
        u2 = self.ups(h2, size)
        u4 = self.ups(h4, size)
        u8 = self.ups(h8, size)
        u16 = self.ups(h16, size)
        uu = self.rb(torch.cat([u2, u4, u8, u16], dim=1))

        out = self.conv_out(s2 + uu)
        return out
