
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DRNet(nn.Module):
    def __init__(self, in_channels=1, depth_rate=32, num_dense_layers=5):
        super(DRNet, self).__init__()

        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(depth_rate, in_channels, kernel_size=3, padding=1)

        self.nbc0 = NBC(in_channels=depth_rate, num_of_layers=num_dense_layers)
        self.nbc2 = NBC(in_channels=depth_rate, num_of_layers=num_dense_layers)
        self.nbc4 = NBC(in_channels=depth_rate, num_of_layers=num_dense_layers)
        self.nbc8 = NBC(in_channels=depth_rate, num_of_layers=num_dense_layers)
        self.nbc16 = NBC(in_channels=depth_rate, num_of_layers=num_dense_layers)

        self.conv_mid = nn.Conv2d(in_channels=5*depth_rate, out_channels=depth_rate, kernel_size=1, padding=0)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def ups(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):

        size = x.size()[2:]
        s0 = self.conv_in(x)

        p2 = self.pool(s0, 2)
        p4 = self.pool(s0, 4)
        p8 = self.pool(s0, 8)
        p16 = self.pool(s0, 16)

        y0 = self.nbc0(s0)
        y2 = self.ups(self.nbc2(p2), size)
        y4 = self.ups(self.nbc4(p4), size)
        y8 = self.ups(self.nbc8(p8), size)
        y16 = self.ups(self.nbc16(p16), size)

        y = self.conv_mid(torch.cat([y0, y2, y4, y8, y16], dim=1))
        out = self.conv_out(y)

        return out
