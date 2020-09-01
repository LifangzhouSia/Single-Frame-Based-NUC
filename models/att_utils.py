import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Channel-wise Attention ---#
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, ch_in, reduction_ratio=4, pool_types=['avg']):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(ch_in, ch_in // reduction_ratio),
            nn.ReLU(),
            nn.Linear(ch_in // reduction_ratio, ch_in)
        )
        self.pool_types = pool_types
        self.ch_in = ch_in

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand(-1, self.ch_in, x.size(2), x.size(3))
        return scale


#--- Spatial-wise Attention ---#
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_compress = self.compress(x)
        scale = self.spatial(x_compress)
        return scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        c_att = self.ChannelGate(x)
        s_att = self.SpatialGate(x)
        att   = c_att * s_att
        return att


# class residual_attetion_merging(nn.Module):
#     def __init__(self):
#         self.att = AttentionBlock()
#         self.denseblock = DB()
#     def forward(self, x, h, b):
#         a = attention(x)
#         b = b / (gamma*b+1)
#         g = self.denseblock(x)
#         k = a / (a + b)
#         h = k*g + (1-k)*h
#         out = x + h
#         return out, h, b





























