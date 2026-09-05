import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "scale_mae"))

try:
    import models_vit
except ImportError:
    pass

class DoubleConv(nn.Module):
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
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ViTAdapter(nn.Module):
    def __init__(self, vit_embed_dim=768):
        super().__init__()
        self.stride_16_conv = nn.Conv2d(vit_embed_dim, 256, 1)
        self.stride_32_conv = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.stride_8_up = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.stride_4_up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.stride_2_up = nn.ConvTranspose2d(64, 64, 2, stride=2)
        
    def forward(self, x_vit, H_img, W_img):
        B, N, C = x_vit.shape
        x_vit = x_vit[:, 1:, :] 
        
        h = H_img // 16
        w = W_img // 16
        
        x_16 = x_vit.transpose(1, 2).reshape(B, C, h, w)
        x_16 = self.stride_16_conv(x_16)
        
        x_32 = self.stride_32_conv(x_16)
        x_8  = self.stride_8_up(x_16)
        x_4  = self.stride_4_up(x_8)
        x_2  = self.stride_2_up(x_4)
        
        dummy_img = torch.zeros(B, 3, H_img, W_img, device=x_16.device)
        return [dummy_img, x_2, x_4, x_8, x_16, x_32]


class SiameseScaleMAE(nn.Module):
    def __init__(self, num_damage_classes=5, vit_model="vit_base_patch16", input_res=1.0):
        super().__init__()
        self.input_res = input_res
        
        try:
            if vit_model == "vit_base_patch16":
                self.encoder = models_vit.vit_base_patch16(num_classes=0, drop_path_rate=0.1, img_size=512)
                embed_dim = 768
            else:
                self.encoder = models_vit.vit_large_patch16(num_classes=0, drop_path_rate=0.1, img_size=512)
                embed_dim = 1024
        except Exception as e:
            import traceback; traceback.print_exc()
            self.encoder = None
            embed_dim = 768
            
        self.adapter = ViTAdapter(vit_embed_dim=embed_dim)
        encoder_channels = [3, 64, 64, 128, 256, 512]
        self.bottleneck = DoubleConv(encoder_channels[-1] * 2, 512)

        skip_chs = [c * 2 for c in encoder_channels[1:-1]]
        skip_chs = list(reversed(skip_chs))
        dec_out_chs = [256, 128, 64, 32]

        self.decoder = nn.ModuleList()
        in_ch = 512
        for sk, out in zip(skip_chs, dec_out_chs):
            self.decoder.append(DecoderBlock(in_ch, sk, out))
            in_ch = out

        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = DoubleConv(32, 32)
        self.loc_head = nn.Conv2d(32, 2, 1)
        self.dmg_head = nn.Conv2d(32, num_damage_classes, 1)

    def forward_single(self, x):
        B, C, H, W = x.shape
        res_tensor = torch.full((B, 1), self.input_res, device=x.device, dtype=torch.float32)
        
        if self.encoder is not None:
            x_tokens = self.encoder.forward_features(x, input_res=res_tensor)
        else:
            N = (H // 16) * (W // 16) + 1
            x_tokens = torch.zeros(B, N, 768, device=x.device)
            
        feats = self.adapter(x_tokens, H, W)
        return feats

    def forward(self, pre_img, post_img):
        pre_feats  = self.forward_single(pre_img)
        post_feats = self.forward_single(post_img)

        x = torch.cat([pre_feats[-1], post_feats[-1]], dim=1)
        x = self.bottleneck(x)

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

if __name__ == "__main__":
    model = SiameseScaleMAE()
    pre = torch.randn(2, 3, 512, 512)
    post = torch.randn(2, 3, 512, 512)
    loc, dmg = model(pre, post)
    print(f"Loc: {loc.shape}, Dmg: {dmg.shape}")
