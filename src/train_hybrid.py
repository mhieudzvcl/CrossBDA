"""
train_hybrid.py - Phase 3.3: Hybrid FDA Augmentation + Few-shot Fine-tuning
Strategy:
  - Start from fda_best_model.pth (already color-invariant from FDA training)
  - Freeze encoder only to preserve feature extraction
  - Fine-tune bottleneck, decoder, loc_head, dmg_head on 10% ida-BD data
  - Use very low LR to avoid catastrophic forgetting
"""
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, r"H:\KhoaLuan")
from src.model import SiameseUNet


class IdaSplitDataset(Dataset):
    def __init__(self, data_dir):
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.pre_images = sorted(glob.glob(os.path.join(self.img_dir, "*_pre_disaster.png")))

        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={"image2": "image", "mask2": "mask"})

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_path = self.pre_images[idx]
        post_path = pre_path.replace("_pre_", "_post_")
        mask_path = os.path.join(self.mask_dir, os.path.basename(post_path))

        pre_img = np.array(Image.open(pre_path).convert("RGB").resize((512, 512)))
        post_img = np.array(Image.open(post_path).convert("RGB").resize((512, 512)))
        dmg_mask = np.array(Image.open(mask_path).resize((512, 512), Image.NEAREST))
        loc_mask = (dmg_mask > 0).astype(np.uint8)

        t = self.aug(image=pre_img, image2=post_img, mask=loc_mask, mask2=dmg_mask)
        return t["image"], t["image2"], t["mask"].long(), t["mask2"].long()


def train_hybrid(data_dir, out_name, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training hybrid model: {out_name}")

    model = SiameseUNet(encoder_name="resnet34", encoder_weights=None).to(device)
    fda_ckpt = r"H:\KhoaLuan\experiments\fda_results\fda_best_model.pth"
    model.load_state_dict(torch.load(fda_ckpt, map_location=device, weights_only=False)["model_state"])
    print(f"Loaded FDA checkpoint: {fda_ckpt}")

    # Freeze encoder only, unfreeze everything else
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.bottleneck.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    for param in model.final_conv.parameters():
        param.requires_grad = True
    for param in model.loc_head.parameters():
        param.requires_grad = True
    for param in model.dmg_head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M total")

    ds = IdaSplitDataset(data_dir)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    print(f"Training images: {len(ds)}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loc_criterion = nn.CrossEntropyLoss()
    dmg_criterion = nn.CrossEntropyLoss()

    model.train()
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for pre, post, loc, dmg in loader:
            pre, post = pre.to(device), post.to(device)
            loc, dmg = loc.to(device), dmg.to(device)

            optimizer.zero_grad()
            loc_pred, dmg_pred = model(pre, post)

            loss_loc = loc_criterion(loc_pred, loc)
            loss_dmg = dmg_criterion(dmg_pred, dmg)
            loss = 0.4 * loss_loc + 0.6 * loss_dmg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if epoch % 5 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch:02d}/{epochs} Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.2e}")

    out_dir = r"H:\KhoaLuan\experiments\hybrid_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    torch.save({"model_state": model.state_dict()}, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    train_hybrid(
        data_dir=r"H:\KhoaLuan\data\ida-BD\split\train_10",
        out_name="hybrid_fs10_model.pth",
        epochs=30
    )