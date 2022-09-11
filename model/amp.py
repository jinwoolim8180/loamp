import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import BasicBlock, ResidualBlock, RNNCell


class LOAMP(nn.Module):
    """
    Approximate message passing with learned onsager term.
    The entire module.
    """
    def __init__(self, in_channels, cs_channels, n_channels, stages, scale):
        super(LOAMP, self).__init__()
        self.in_channels = in_channels
        self.cs_channels = cs_channels
        self.n_channels = n_channels
        self.stages = stages
        self.scale = scale

        self.measurement = nn.Parameter(torch.randn(cs_channels, scale * scale * in_channels),
                                        requires_grad=False)
        self.transpose = self.measurement.t().contiguous().view(scale * scale, cs_channels, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)
        self.eta = nn.ModuleList([])
        for i in range(self.stages):
            self.eta.append(
                nn.Sequential(
                    BasicBlock(cs_channels, cs_channels, kernel_size=1, norm=False)
                )
            )

    def forward(self, x):
        phi = self.measurement.contiguous()\
            .view(self.cs_channels, self.in_channels, self.scale, self.scale)
        y = F.conv2d(x, phi, stride=self.scale)
        out = self.shuffle(F.conv2d(y, self.transpose.to(y.device)))
        z = torch.zeros_like(y).to(x.device)
        for i in range(self.stages):
            # z *= self.cs_channels / (self.cs_channels * self.scale * self.scale)
            z = y - F.conv2d(x, phi, stride=self.scale)
            z = z + self.eta[0](z)
            out = self.shuffle(F.conv2d(z, self.transpose.to(z.device))) + x
            # out = out + self.eta[0](out)
        return out