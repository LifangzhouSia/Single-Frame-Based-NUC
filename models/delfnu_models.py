
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_block import CBAM
from models.nonlocal_block import NonLocalBlock


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
        return self.bc(x)


class NLSPP(nn.Module):
    def __init__(self, in_channels):
        super(NLSPP, self).__init__()
        self.conv_out = nn.Conv2d(4*in_channels, in_channels, kernel_size=1, padding=0)

        self.conv2 = basicConv(in_channels)
        self.conv4 = basicConv(in_channels)
        self.conv8 = basicConv(in_channels)
        self.conv16 = basicConv(in_channels)

        # self.nlb2 = NonLocalBlock(in_channels)
        # self.nlb4 = NonLocalBlock(in_channels)
        # self.nlb8 = NonLocalBlock(in_channels)
        # self.nlb16 = NonLocalBlock(in_channels)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def ups(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        p2  = self.ups(self.conv2(self.pool(x, 2)), size)
        p4  = self.ups(self.conv4(self.pool(x, 4)), size)
        p8  = self.ups(self.conv8(self.pool(x, 8)), size)
        p16 = self.ups(self.conv16(self.pool(x, 16)), size)
        s   = torch.cat([p2, p4, p8, p16], dim=1)
        out = self.conv_out(s)
        return out


class rdp_nuc(nn.Module):
    def __init__(self, in_channels=1, depth_rate=16, kernel_size=3, num_dense_layer=4, growth_rate=16):
        super(rdp_nuc, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=1)
        self.conv1 = nn.Conv2d(depth_rate, depth_rate, kernel_size=kernel_size, padding=1)

        self.db1 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db2 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db3 = DB(depth_rate, num_dense_layer, growth_rate)

        self.spp = NLSPP(depth_rate)

        self.att = nn.Sequential(
            nn.Conv2d(depth_rate, depth_rate, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_rate, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.gff_1x1 = nn.Conv2d(depth_rate*3, depth_rate, kernel_size=1, padding=0)

        self.regressk = nn.Sequential(
            nn.Conv2d(depth_rate, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.regressb = nn.Sequential(
            nn.Conv2d(depth_rate, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.regressk = nn.Conv2d(depth_rate, 1, kernel_size=3, padding=1)
        self.regressb = nn.Conv2d(depth_rate, 1, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        s = self.conv_in(x)
        s1 = self.conv1(s)
        d1 = self.db1(s1)
        d2 = self.db2(d1)
        d3 = self.db3(d2)
        dc = torch.cat([d1, d2, d3], dim=1)
        s2 = self.gff_1x1(dc)

        # w  = self.att(s2)
        s3 = self.spp(s2)
        b = self.conv_out(s3)
        out = x - b

        # k = self.regressk(s3)
        # b = self.regressb(s3)
        #
        # outk = k*x
        # outb = x+b
        # out = w*outk + (1-w)*outb

        return out