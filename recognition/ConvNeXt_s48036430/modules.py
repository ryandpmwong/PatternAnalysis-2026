import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in the range [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep_prob


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, drop_path_rate=0.0):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.activation = nn.GELU()
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.layer_scale * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)
