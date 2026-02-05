import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Two convolutions + batch norm + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric image synthesis.
    """
    def __init__(self, in_channels=2, out_channels=1, features=[32, 64, 128]):
        super().__init__()
        
        # Encoder
        self.encoder1 = ConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(2)
        
        self.encoder2 = ConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[1], features[2])
        
        # Decoder
        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(features[1]*2, features[1])
        
        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(features[0]*2, features[0])
        
        # Output
        self.final = nn.Conv3d(features[0], out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        
        # Bottleneck
        b = self.bottleneck(self.pool2(e2))
        
        # Decoder + skip connections
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        return self.final(d1)


if __name__ == "__main__":
    model = UNet3D(in_channels=2, out_channels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"3D U-Net parameters: {n_params:,}")
    
    # Test
    x = torch.randn(1, 2, 64, 64, 64)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")