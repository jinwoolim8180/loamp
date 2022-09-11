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
        self.n_channels = n_channels
        self.stages = stages
        self.scale = scale

        self.measurement = nn.Parameter(
            torch.randn(cs_channels, scale * scale * in_channels)
            .contiguous().view(cs_channels, in_channels, scale, scale), requires_grad=False)
        self.transpose = nn.Sequential(
            nn.Conv2d(cs_channels, in_channels * scale * scale, kernel_size=1),
            nn.PixelShuffle(scale)
        )
        self.eta = nn.Sequential(
            BasicBlock(in_channels, n_channels),
            ResidualBlock(n_channels),
            BasicBlock(n_channels, in_channels)
        )
        self.onsager = RNNCell(cs_channels)

    def forward(self, x):
        y = F.conv2d(x, self.measurement, stride=self.scale)
        out = self.transpose(y)
        z = torch.zeros_like(y).to(x.device)
        h = torch.zeros_like(y).to(x.device)
        for i in range(self.stages):
            z = y - F.conv2d(x, self.measurement, stride=self.scale)
            h = self.onsager(z, h)
            z += h
            out = self.eta(self.transpose(z) + x)
        return out


class AMP_Stage(nn.Module):
    def __init__(self, in_channels, cs_channels, n_channels, scale=2, lamda=1):
        super(AMP_Stage, self).__init__()
        self.in_channels = in_channels
        self.n_channels = n_channels
        self.scale = scale
        self.lamda = lamda
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.transpose = nn.Sequential(
            nn.Conv2d(cs_channels, in_channels * scale * scale, kernel_size=1),
            nn.PixelShuffle(scale)
        )
        self.eta = nn.Sequential(
            BasicBlock(in_channels, n_channels),
            ResidualBlock(n_channels),
            ResidualBlock(n_channels),
            BasicBlock(n_channels, in_channels)
        )
        self.onsager = RNNCell(cs_channels)
        self.basis = nn.Conv2d(cs_channels, cs_channels, kernel_size=1, bias=False)

    def forward(self, x, y, h, measurement):
        z = y - F.conv2d(x, measurement, stride=self.scale)
        # h_t = self.onsager(z, h)
        # z += self.basis(h_t)
        h_t = None
        out = self.eta(self.alpha.unsqueeze(1) * self.transpose(z) + x)
        return out, h_t