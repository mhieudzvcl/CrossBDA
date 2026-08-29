"""
model.py - Siamese U-Net for Building Damage Assessment
Architecture:
  - Shared ResNet-34 encoder (pretrained ImageNet)
  - Two streams: pre & post disaster images
  - Feature fusion via concatenation at each decoder level
  - Two heads:
      * loc_head  : building localization (binary, 2 classes)
      * dmg_head  : damage classification (5 classes: 0=bg, 1-4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """Conv-BN-ReLU x2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """Upsample + fused skip connection"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Pad if sizes do not match (happens with odd input sizes)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net with shared ResNet-34 encoder.
    Input : pre_img, post_img  -- (B, 3, H, W)
    Output: loc_logits (B, 2, H, W), dmg_logits (B, 5, H, W)
    """

    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        num_damage_classes: int = 5,
    ):
        super().__init__()

        # Shared Encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        encoder_channels = self.encoder.out_channels

        # Bottleneck
        self.bottleneck = DoubleConv(encoder_channels[-1] * 2, 512)

        # Decoder (shared, fused features)
        skip_chs = [c * 2 for c in encoder_channels[1:-1]]
        skip_chs = list(reversed(skip_chs))

        dec_out_chs = [256, 128, 64, 32]

        self.decoder = nn.ModuleList()
        in_ch = 512
        for i, (sk, out) in enumerate(zip(skip_chs, dec_out_chs)):
            self.decoder.append(DecoderBlock(in_ch, sk, out))
            in_ch = out

        # Final upsample
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = DoubleConv(32, 32)

        # Output Heads
        self.loc_head = nn.Conv2d(32, 2, 1)
        self.dmg_head = nn.Conv2d(32, num_damage_classes, 1)

    def forward_single(self, x):
        """Encode a single image, returning a list of features."""
        return self.encoder(x)

    def forward(self, pre_img, post_img):
        # Encode
        pre_feats  = self.forward_single(pre_img)
        post_feats = self.forward_single(post_img)

        # Bottleneck
        x = torch.cat([pre_feats[-1], post_feats[-1]], dim=1)
        x = self.bottleneck(x)

        # Decode with fused skip connections
        skips = [
            torch.cat([pre_feats[i], post_feats[i]], dim=1)
            for i in range(len(pre_feats) - 2, 0, -1)
        ]

        for block, skip in zip(self.decoder, skips):
            x = block(x, skip)

        x = self.final_up(x)
        x = self.final_conv(x)

        loc_logits = self.loc_head(x)
        dmg_logits = self.dmg_head(x)

        return loc_logits, dmg_logits


# Quick sanity check
if __name__ == '__main__':
    model = SiameseUNet()
    pre  = torch.randn(2, 3, 512, 512)
    post = torch.randn(2, 3, 512, 512)
    loc, dmg = model(pre, post)
    print(f'loc: {loc.shape}')
    print(f'dmg: {dmg.shape}')
    print('Model OK!')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params/1e6:.1f}M')