import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== CBAM Module ===== #
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ===== Residual Dense Block ===== #
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(
                nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1)
            )
            channels += growth_rate
        self.local_fusion = nn.Conv2d(channels, in_channels, kernel_size=1)

    def forward(self, x):
        features = [x]
        for conv in self.layers:
            out = F.relu(conv(torch.cat(features, dim=1)))
            features.append(out)
        fused = self.local_fusion(torch.cat(features, dim=1))
        return fused + x  # Residual connection

# ===== CBAM-RDB-UNet ===== #
class CBAM_RDB_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, kernel_size=3):
        super().__init__()
        k = kernel_size

        # Initial convolution
        self.initial = nn.Conv2d(in_channels, base_channels, kernel_size=k, padding=k//2)

        # Encoder
        self.enc1 = nn.Sequential(
            ResidualDenseBlock(base_channels, k),
            CBAM(base_channels)
        )
        self.enc2 = nn.Sequential(
            ResidualDenseBlock(base_channels * 2, k),
            CBAM(base_channels * 2)
        )
        self.enc3 = nn.Sequential(
            ResidualDenseBlock(base_channels * 4, k),
            CBAM(base_channels * 4)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualDenseBlock(base_channels * 8, k),
            CBAM(base_channels * 8)
        )

        # Downsampling
        self.pool = nn.MaxPool2d(2)

        # Upsampling
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)

        # Decoder
        self.dec3 = nn.Sequential(
            ResidualDenseBlock(base_channels * 8, k),
            CBAM(base_channels * 8)
        )
        self.dec2 = nn.Sequential(
            ResidualDenseBlock(base_channels * 4, k),
            CBAM(base_channels * 4)
        )
        self.dec1 = nn.Sequential(
            ResidualDenseBlock(base_channels * 2, k),
            CBAM(base_channels * 2)
        )

        # Output layer
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.final(d1)
        return out  # Optionally wrap with torch.sigmoid(out) if needed


