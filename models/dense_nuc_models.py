
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.att_utils import CBAM


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


class groupDenseBlock(nn.Module):
    def __init__(self, depth_rate, num_dense_layer, growth_rate):
        super(groupDenseBlock, self).__init__()
        self.db1 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db2 = DB(depth_rate, num_dense_layer, growth_rate)
        self.db3 = DB(depth_rate, num_dense_layer, growth_rate)
        self.gff_1x1 = nn.Conv2d(depth_rate * 3, depth_rate, kernel_size=1, padding=0)

    def forward(self, x):
        d1 = self.db1(x)
        d2 = self.db2(d1)
        d3 = self.db3(d2)
        dc = torch.cat([d1, d2, d3], dim=1)
        out = self.gff_1x1(dc)
        out = out + x
        return out


class BasicBlock(nn.Module):
    def __init__(self, depth_rate=16, num_dense_layer=4, growth_rate=16):
        super(BasicBlock, self).__init__()
        self.gdb = groupDenseBlock(depth_rate, num_dense_layer, growth_rate)
        self.att = CBAM(gate_channels=depth_rate)

    def forward(self, x):
        g = self.gdb(x)
        a = self.att(x)
        out = a*g
        return out


class dense_nuc_model(nn.Module):
    def __init__(self, in_channels=1, depth_rate=16, kernel_size=3, num_dense_layer=4, growth_rate=16):
        super(dense_nuc_model, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2)

        self.bb0 = BasicBlock(depth_rate, num_dense_layer, growth_rate)
        self.bb1 = BasicBlock(depth_rate, num_dense_layer, growth_rate)
        self.bb2 = BasicBlock(depth_rate, num_dense_layer, growth_rate)
        self.bb3 = BasicBlock(depth_rate, num_dense_layer, growth_rate)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def ups(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, img):

        size = img.size()[2:]
        s = self.conv_in(img)

        h0 = self.bb0(s)
        s1 = s + h0

        p1 = self.pool(s1, 4)
        h1 = self.bb1(p1)
        s2 = self.ups(p1 + h1, size)

        p2 = self.pool(s2, 16)
        h2 = self.bb2(p2)
        s3 = self.ups(p2 + h2, size)

        p3 = self.pool(s3, 64)
        h3 = self.bb3(p3)
        s4 = self.ups(p3 + h3, size)

        out = self.conv_out(s4)

        return out
