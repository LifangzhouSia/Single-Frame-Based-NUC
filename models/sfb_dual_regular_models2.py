
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NBC(nn.Module):
    def __init__(self, in_channels, num_of_layers):
        super(NBC, self).__init__()
        modules = []
        for i in range(num_of_layers):
            modules.append(basicConv(in_channels))
        self.nbc = nn.Sequential(*modules)

    def forward(self, x):
        out = self.nbc(x)
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


class DRNet(nn.Module):
    def __init__(self, in_channels=1, depth_rate=16, num_dense_layer=4, growth_rate=16):
        super(DRNet, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=3, padding=1)

        self.db0 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db4 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db16 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db64 = DB(depth_rate, num_dense_layer, growth_rate)

        self.att0 = basicAttention(depth_rate, depth_rate)
        self.att4 = basicAttention(depth_rate, depth_rate)
        self.att16 = basicAttention(depth_rate, depth_rate)
        self.att64 = basicAttention(depth_rate, depth_rate)

        self.conv_mid = nn.Conv2d(in_channels=4*depth_rate, out_channels=depth_rate, kernel_size=1, padding=0)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def ups(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):

        size = x.size()[2:]
        s0 = self.conv_in(x)

        p4 = self.pool(s0, 4)
        p16 = self.pool(s0, 16)
        p64 = self.pool(s0, 64)

        a0 = self.att0(s0)
        a4 = self.att4(p4)
        a16 = self.att16(p16)
        a64 = self.att64(p64)

        y0 = s0 + a0*self.db0(s0)
        y4 = self.ups(p4 + a4*self.db4(p4), size)
        y16 = self.ups(p16 + a16*self.db16(p16), size)
        y64 = self.ups(p64 + a64*self.db64(p64), size)

        y = self.conv_mid(torch.cat([y0, y4, y16, y64], dim=1))
        out = self.conv_out(y)

        return out
