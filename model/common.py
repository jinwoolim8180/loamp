import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic block with no residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, norm=True):
        super(BasicBlock, self).__init__()
        if norm:
            self.module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.module(x)


class ResidualBlock(nn.Module):
    """
    Convolution block with residual connection
    """
    def __init__(self, n_channels, kernel_size=3, norm=True):
        super(ResidualBlock, self).__init__()
        self.module = nn.Sequential(
            BasicBlock(n_channels, n_channels, kernel_size=kernel_size, norm=norm),
            BasicBlock(n_channels, n_channels, kernel_size=kernel_size, norm=norm)
        )

    def forward(self, x):
        return x + self.module(x)


class RNNCell(nn.Module):
    """
    RNN Cell for Onsager term
    """
    def __init__(self, n_channels):
        super(RNNCell, self).__init__()
        self.W = nn.Conv2d(2 * n_channels, n_channels, kernel_size=1)

    def forward(self, x, h):
        return F.tanh(self.W(torch.cat([x, h], dim=1)))