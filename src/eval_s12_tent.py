"""
eval_s12_tent.py - Test-Time Entropy Minimization on S12 Dataset
"""
import os, sys, glob, warnings, yaml
import numpy as np
import torch
import cv2
import tifffile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.factory import create_model
from src.metrics import MetricAccumulator, DAMAGE_CLASS_NAMES
from src.tent import TENT

class S12Dataset(Dataset):
    def __init__(self, s12_dir, xbd_dir):
        self.s12_dir = s12_dir
        self.xbd_dir = xbd_dir
        self.pre_images = sorted(glob.glob(os.path.join(s12_dir, "*_pre_disaster_s2_tci.tif")))
        
        self.mask_lookup = {}
        for split in ['train', 'test', 'hold', 'tier3']:
            mask_dir = os.path.join(xbd_dir, split, "targets")
            if os.path.exists(mask_dir):
                for f in os.listdir(mask_dir):
                    if f.endswith('_post_disaster_target.png'):
                        self.mask_lookup[f] = os.path.join(mask_dir, f)
                        
        self.valid_pairs = []
        for pre in self.pre_images:
            base_name = os.path.basename(pre).replace("_pre_disaster_s2_tci.tif", "")
            mask_name = f"{base_name}_post_disaster_target.png"
            if mask_name in self.mask_lookup:
                post = pre.replace("_pre_", "_post_")
                if os.path.exists(post):
                    self.valid_pairs.append({
                        'pre': pre,
                        'post': post,
                        'mask': self.mask_lookup[mask_name]
                    })
        print(f"Found {len(self.valid_pairs)} matching S12-mask pairs to evaluate.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        pre_img = tifffile.imread(pair['pre'])
        post_img = tifffile.imread(pair['post'])
        pre_img_1024 = cv2.resize(pre_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        post_img_1024 = cv2.resize(post_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        
        pre_tensor = torch.from_numpy(pre_img_1024.transpose(2, 0, 1)).float() / 255.0
        post_tensor = torch.from_numpy(post_img_1024.transpose(2, 0, 1)).float() / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pre_tensor = (pre_tensor - mean) / std
        post_tensor = (post_tensor - mean) / std
        
        mask = Image.open(pair['mask'])
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return pre_tensor, post_tensor, mask_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Duong dan den file .pth")
    parser.add_argument("--config", type=str, required=True, help="Duong dan den file .yaml")
    parser.add_argument("--data_dir_s12", type=str, default=r"H:\KhoaLuan\data\xBD-S12\s2_tci")
    parser.add_argument("--data_dir_xbd", type=str, default=r"H:\KhoaLuan\data\xBD")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(args.data_dir_s12):
        args.data_dir_s12 = r"H:\KhoaLuan\scratch\xbd_s12\s2_tci"

    dataset = S12Dataset(s12_dir=args.data_dir_s12, xbd_dir=args.data_dir_xbd)
    if len(dataset) == 0:
        print("Loi: Khong tim thay du lieu S12.")
        return
        
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if 'model' in cfg and 'encoder_weights' in cfg['model']:
        cfg['model']['encoder_weights'] = None

    print(f"Creating model from {args.config}...")
    model = create_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # Wrap model with TENT (This handles setting model.train() and Requires_grad)
    print("\n[TENT] Wrapping model with Test-Time Entropy Minimization...")
    tent_model = TENT(model, lr=1e-3, steps=1)
    
    accumulator = MetricAccumulator()

    print("\n[TENT] Evaluating with online adaptation...")
    for pre_imgs, post_imgs, targets in tqdm(loader, desc="Evaluating TENT"):
        pre_imgs = pre_imgs.to(device)
        post_imgs = post_imgs.to(device)
        targets = targets.to(device)

        # TENT forward handles the backward pass inside
        with torch.amp.autocast(device.type):
            out_loc, out_dmg = tent_model(pre_imgs, post_imgs)

        loc_targets = (targets > 0).long()
        dmg_targets = targets.long()
        accumulator.update(out_loc, out_dmg, loc_targets, dmg_targets)
        
        # Reset parameters optionally after each image if you want episodic TENT, 
        # or keep them updated for sequential TENT. Here we use sequential (default TENT)

    metrics = accumulator.compute()
    
    print("\nEVALUATION RESULTS ON xBD-S12 (TENT)")
    print(f"xView2 Score     : {metrics['xview2_score']:.4f}")
    print(f"F1 Localization  : {metrics['f1_loc']:.4f}")
    print(f"F1 Damage (macro): {metrics['f1_dmg_macro']:.4f}")
    for k in [1, 2, 3, 4]:
        class_name = DAMAGE_CLASS_NAMES[k]
        print(f"  F1 {class_name:<14}: {metrics[f'f1_{class_name}']:.4f}")

if __name__ == "__main__":
    main()
