"""
train_pseudolabel.py - Fine-tune on ida-BD using Pseudo-Labels (Self-Training)
"""
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import sys

sys.path.insert(0, r"H:\KhoaLuan")
from src.model import SiameseUNet
from src.losses import CombinedLoss

class PseudoLabelDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.pre_images = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.endswith("_pre_disaster.png"):
                    self.pre_images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_path = self.pre_images[idx]
        post_path = pre_path.replace("_pre_disaster.png", "_post_disaster.png")
        
        # Load mask
        prefix = os.path.basename(pre_path).replace("_pre_disaster.png", "")
        mask_path = os.path.join(self.mask_dir, prefix + "_target.png")
        
        pre_img = np.array(Image.open(pre_path).convert("RGB"))
        post_img = np.array(Image.open(post_path).convert("RGB"))
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
        else:
            mask = np.zeros((pre_img.shape[0], pre_img.shape[1]), dtype=np.uint8)

        loc_mask = (mask > 0).astype(np.uint8)
        dmg_mask = mask.copy()

        if self.transform:
            augmented = self.transform(image=pre_img, post_image=post_img, loc_mask=loc_mask, dmg_mask=dmg_mask)
            pre_img = augmented['image']
            post_img = augmented['post_image']
            loc_mask = augmented['loc_mask']
            dmg_mask = augmented['dmg_mask']

        return pre_img, post_img, loc_mask.long(), dmg_mask.long()

def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'post_image': 'image', 'loc_mask': 'mask', 'dmg_mask': 'mask'})

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load fda_best_model
    model = SiameseUNet(encoder_name="resnet34", encoder_weights=None).to(device)
    ckpt_path = r"H:\KhoaLuan\experiments\fda_results\fda_best_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])

    # FREEZE ENCODER to prevent catastrophic forgetting
    for param in model.encoder.parameters():
        param.requires_grad = False

    img_dir = r"H:\KhoaLuan\data\ida-BD\PRJ-3563"
    mask_dir = r"H:\KhoaLuan\data\ida-BD\pseudo_labels"
    
    dataset = PseudoLabelDataset(img_dir, mask_dir, transform=get_train_transform())
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    print(f"Training on {len(dataset)} pseudo-labeled images.")

    criterion = CombinedLoss(w_loc=0.3, w_dmg=0.7)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=1e-4)

    epochs = 20
    model.train()
    
    out_dir = r"H:\KhoaLuan\experiments\pseudolabel_results"
    os.makedirs(out_dir, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for pre, post, loc_true, dmg_true in pbar:
            pre, post = pre.to(device), post.to(device)
            loc_true, dmg_true = loc_true.to(device), dmg_true.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type):
                loc_pred, dmg_pred = model(pre, post)
                loss, _ = criterion(loc_pred, dmg_pred, loc_true, dmg_true)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch} Avg Loss: {epoch_loss/len(loader):.4f}")

    save_path = os.path.join(out_dir, "pseudo_model.pth")
    torch.save({"model_state": model.state_dict()}, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()