import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch import autograd
import torch.nn.functional as F


class Dconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Dconv, self).__init__()
        self.dc = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, t):
        out = self.dc(t)
        return out


class SFB_DCUnet(nn.Module):
    def __init__(self, ch):
        super(SFB_DCUnet, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ups = nn.Upsample(scale_factor=2)

        self.conv_in = nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1)

        # --- 1st Stage --- #
        self.dc0 = Dconv(ch, ch)
        self.dc1 = Dconv(ch, ch)
        self.dc2 = Dconv(ch, ch)
        self.dc3 = Dconv(ch, ch)
        self.dc4 = Dconv(ch, ch)
        self.dc44 = Dconv(ch, ch)
        self.uc3 = Dconv(2 * ch, ch)
        self.uc2 = Dconv(2 * ch, ch)
        self.uc1 = Dconv(2 * ch, ch)
        self.uc0 = Dconv(2 * ch, ch)

    def forward(self, img):

        x0 = self.conv_in(img)

        h0 = self.dc0(x0)
        h1 = self.dc1(self.pool(h0))
        h2 = self.dc2(self.pool(h1))
        h3 = self.dc3(self.pool(h2))
        h4 = self.dc4(self.pool(h3))
        d4 = self.dc44(h4)
        d3 = self.uc3(torch.cat([h3, self.ups(d4)], dim=1))
        d2 = self.uc2(torch.cat([h2, self.ups(d3)], dim=1))
        d1 = self.uc1(torch.cat([h1, self.ups(d2)], dim=1))
        d0 = self.uc0(torch.cat([h0, self.ups(d1)], dim=1))

        n = self.conv_out(d0)
        out = img - n

        return out