
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

    def forward(self, x):

        size = x.size()[2:]
        s1 = self.conv_in(x)

        # --- Dense block --- #
        d1 = self.db1(s1)
        d2 = self.db2(d1)
        d3 = self.db3(d2)
        dc = torch.cat([d1, d2, d3], dim=1)
        s2 = self.gff_1x1(dc)

        # --- Pyramid block --- #
        p2 = self.pool(s2, 2)
        p4 = self.pool(s2, 4)
        p8 = self.pool(s2, 8)
        p16 = self.pool(s2, 16)

        # --- Pyramid convolution block --- #
        u2 = self.ups(self.bc2(p2), size)
        u4 = self.ups(self.bc4(p4), size)
        u8 = self.ups(self.bc8(p8), size)
        u16 = self.ups(self.bc16(p16), size)
        uu = self.rb(torch.cat([u2, u4, u8, u16], dim=1))
        s3 = s2 + uu

        s3 = s3 + self.bc0(s3)
        out = self.conv_out(s3)
        return out
