import torch
import torch.nn as nn
import numpy as np
from torch import autograd
import torch.nn.functional as F

class VerticalNLBlock(nn.Module):
    def __init__(self):
        super(VerticalNLBlock, self).__init__()

    def vectorBlock(self, x):
        x_phi = x.contiguous()
        x_theta = x.permute(0, 2, 1).contiguous()
        x_g = x.permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = F.softmax(mul_theta_phi, dim=1)
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous()
        return mul_theta_phi_g

    def forward(self, x):
        b, c, h, w = x.size()
        for k in range(w):
            x[:, :, :, k] = self.vectorBlock(x[:, :, :, k])
        return x