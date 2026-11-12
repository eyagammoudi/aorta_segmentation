"""3D U-Net (Çiçek et al., 2016) with four resolution levels."""
import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """(Conv3d -> BatchNorm -> ReLU) x 2."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """Returns logits; apply a sigmoid for probabilities."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: int = 16):
        super().__init__()
        f = [features * 2 ** i for i in range(5)]
        self.encoders = nn.ModuleList([DoubleConv3D(in_channels, f[0])] +
                                      [DoubleConv3D(f[i], f[i + 1]) for i in range(3)])
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(f[3], f[4])
        self.upconvs = nn.ModuleList([nn.ConvTranspose3d(f[i + 1], f[i], 2, stride=2) for i in reversed(range(4))])
        self.decoders = nn.ModuleList([DoubleConv3D(f[i] * 2, f[i]) for i in reversed(range(4))])
        self.head = nn.Conv3d(f[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.head(x)
