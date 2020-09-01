
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Upsampling block in GridDehazeNet  --- #
class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv, self).__init__()
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.c(x))
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
        p2  = self.ups(self.pool(x, 2), size)
        p4  = self.ups(self.pool(x, 4), size)
        p8  = self.ups(self.pool(x, 8), size)
        p16 = self.ups(self.pool(x, 16), size)
        s   = torch.cat([x, p2, p4, p8, p16], dim=1)
        out = self.conv1(s)
        return out


class rdp_nuc(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3):
        super(rdp_nuc, self).__init__()

        self.conv_in = conv(1, 32)
        self.conv1   = conv(32, 32)
        self.conv2   = conv(32, 32)
        self.conv3   = conv(32, 32)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.spp1 = SPP(in_channels=32)
        self.spp2 = SPP(in_channels=32)
        self.spp3 = SPP(in_channels=32)

    def forward(self, x):

        s0 = self.conv_in(x)
        p1 = self.spp1(s0)
        s1 = self.conv1(p1)
        p2 = self.spp2(s1)
        s2 = self.conv2(p2)
        p3 = self.spp3(s2)
        s3 = self.conv3(p3)

        b = self.conv_out(s3)
        out = x - b

        return out
