import os
import sys
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from shapely import wkt
from shapely.geometry import mapping
from PIL import ImageDraw

sys.path.insert(0, r'H:\KhoaLuan')
from src.model import SiameseUNet
from src.metrics import MetricAccumulator

def _parse_damage_mask(json_path, out_size=512):
    label_map = {'no-damage': 1, 'minor-damage': 2, 'major-damage': 3, 'destroyed': 4, 'un-classified': 1}
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    try:
        with open(json_path) as f:
            data = json.load(f)
        for feat in data.get('features', {}).get('xy', []):
            props = feat.get('properties', {})
            cls   = label_map.get(props.get('subtype', ''), 0)
            geom  = wkt.loads(feat['wkt'])
            coords = list(mapping(geom)['coordinates'][0])
            if len(coords) >= 3:
                flat = [(float(x), float(y)) for x, y in coords]
                img_tmp = Image.fromarray(mask)
                ImageDraw.Draw(img_tmp).polygon(flat, fill=int(cls))
                mask = np.array(img_tmp)
    except Exception:
        pass
    mask = np.array(Image.fromarray(mask).resize((out_size, out_size), Image.NEAREST))
    return mask

class FewShotDataset(Dataset):
    def __init__(self, data_dir, is_train=True):
        self.img_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        self.pre_images = sorted(glob.glob(os.path.join(self.img_dir, '*_pre_disaster.png')))
        
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={'image2': 'image', 'mask2': 'mask'}) if is_train else None

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_path = self.pre_images[idx]
        post_path = pre_path.replace('_pre_', '_post_')
        lbl_post_p = os.path.join(self.mask_dir, os.path.basename(post_path).replace('.png', '.json'))
        
        pre_img = np.array(Image.open(pre_path).convert('RGB').resize((512, 512)))
        post_img = np.array(Image.open(post_path).convert('RGB').resize((512, 512)))
        
        # parse json if exists, otherwise fallback to PNG
        if os.path.exists(lbl_post_p):
            dmg_mask = _parse_damage_mask(lbl_post_p, 512)
        else:
            png_mask_p = lbl_post_p.replace('.json', '.png')
            dmg_mask = np.array(Image.open(png_mask_p).resize((512, 512), Image.NEAREST))
            
        loc_mask = (dmg_mask > 0).astype(np.uint8)

        if self.aug:
            t = self.aug(image=pre_img, image2=post_img, mask=loc_mask, mask2=dmg_mask)
            pre_t = t['image']
            post_t = t['image2']
            loc_t = t['mask'].long()
            dmg_t = t['mask2'].long()
        else:
            pre_arr  = pre_img.astype(np.float32) / 255.0
            post_arr = post_img.astype(np.float32) / 255.0
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            pre_arr = (pre_arr - mean) / std
            post_arr = (post_arr - mean) / std
            pre_t = torch.from_numpy(pre_arr.transpose(2, 0, 1)).float()
            post_t = torch.from_numpy(post_arr.transpose(2, 0, 1)).float()
            loc_t = torch.from_numpy(loc_mask).long()
            dmg_t = torch.from_numpy(dmg_mask).long()
            
        return pre_t, post_t, loc_t, dmg_t

def train_fewshot(data_dir, out_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device} - {out_name}')
    
    # Load Baseline
    model = SiameseUNet(encoder_name='resnet34', encoder_weights=None).to(device)
    ckpt_path = r'H:\KhoaLuan\experiments\baseline_resnet34\checkpoints\best_model.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
    
    # FREEZE everything to preserve F1_loc
    for param in model.parameters():
        param.requires_grad = False
        
    # Only Dmg_Head requires grad (Linear Probing)
    for param in model.dmg_head.parameters():
        param.requires_grad = True
        
    ds = FewShotDataset(data_dir, is_train=True)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.eval() # Keep BN running stats frozen
    model.dmg_head.train() # Only train dmg_head
    for epoch in range(1, 21):
        for pre, post, loc, dmg in loader:
            pre, post, loc, dmg = pre.to(device), post.to(device), loc.to(device), dmg.to(device)
            optimizer.zero_grad()
            loc_pred, dmg_pred = model(pre, post)
            # Only train dmg_decoder
            loss = criterion(dmg_pred, dmg)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f'Epoch {epoch}/20 Loss: {loss.item():.4f}')
            
    out_path = os.path.join(r'H:\KhoaLuan\experiments\fewshot_results', out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({'model_state': model.state_dict()}, out_path)
    print(f'Saved to {out_path}\n')

if __name__ == '__main__':
    train_fewshot(r'H:\KhoaLuan\data\ida-BD\split\train_5', 'fs5_model.pth')
    train_fewshot(r'H:\KhoaLuan\data\ida-BD\split\train_10', 'fs10_model.pth')