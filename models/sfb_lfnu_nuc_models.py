
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
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        super(RDB, self).__init__()
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
        out = out + x
        return out


class NRDB(nn.Module):
    def __init__(self, depth_rate, num_dense_layer, growth_rate):
        super(NRDB, self).__init__()
        self.conv1 = nn.Conv2d(depth_rate, depth_rate, kernel_size=3, padding=1)
        self.rdb1 = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb2 = RDB(depth_rate, num_dense_layer, growth_rate)
        self.rdb3 = RDB(depth_rate, num_dense_layer, growth_rate)
        self.gff_1x1 = nn.Conv2d(3 * depth_rate, depth_rate, kernel_size=1, padding=0)
        self.gff_3x3 = nn.Conv2d(depth_rate, depth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        s = self.conv1(x)
        f1 = self.rdb1(s)
        f2 = self.rdb2(f1)
        f3 = self.rdb3(f2)
        ff = torch.cat((f1, f2, f3), 1)
        gf1 = self.gff_1x1(ff)
        out = self.gff_3x3(gf1)
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
        self.conv_mid = nn.Conv2d(4 * depth_rate, depth_rate, kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=3, padding=1)

        self.nrdb0 = NRDB(depth_rate, num_dense_layer, growth_rate)
        self.nrdb4 = NRDB(depth_rate, num_dense_layer, growth_rate)
        self.nrdb16 = NRDB(depth_rate, num_dense_layer, growth_rate)
        self.nrdb64 = NRDB(depth_rate, num_dense_layer, growth_rate)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def ups(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):

        size = x.size()[2:]
        s = self.conv_in(x)

        p4 = self.pool(s, 4)
        p16 = self.pool(s, 16)
        p64 = self.pool(s, 64)

        y0 = self.nrdb0(s)
        y4 = self.ups(self.nrdb4(p4), size)
        y16 = self.ups(self.nrdb16(p16), size)
        y64 = self.ups(self.nrdb64(p64), size)
        y = self.conv_mid(torch.cat([y0, y4, y16, y64], dim=1))

        out = self.conv_out(y)

        return out
