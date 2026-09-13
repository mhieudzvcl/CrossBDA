"""
eval_tent.py - Evaluate with TENT on ida-BD
"""
import os, sys, glob, yaml, warnings
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore")

ROOT = r"H:\KhoaLuan"
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

from src.models.factory import create_model
from src.metrics import MetricAccumulator, DAMAGE_CLASS_NAMES
from src.tent import TENT

class IdaDataset(Dataset):
    def __init__(self, data_dir):
        self.img_dir  = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.pre_images = sorted(glob.glob(os.path.join(self.img_dir, "*_pre_disaster.png")))

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_path  = self.pre_images[idx]
        post_path = pre_path.replace("_pre_", "_post_")
        mask_path = os.path.join(self.mask_dir, os.path.basename(post_path))

        pre_img  = Image.open(pre_path).convert("RGB").resize((512, 512), Image.BILINEAR)
        post_img = Image.open(post_path).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask     = Image.open(mask_path).resize((512, 512), Image.NEAREST)

        pre_arr  = np.array(pre_img,  dtype=np.float32) / 255.0
        post_arr = np.array(post_img, dtype=np.float32) / 255.0
        mask_arr = np.array(mask)

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        pre_arr  = (pre_arr  - mean) / std
        post_arr = (post_arr - mean) / std

        pre_t  = torch.from_numpy(pre_arr.transpose(2, 0, 1)).float()
        post_t = torch.from_numpy(post_arr.transpose(2, 0, 1)).float()
        loc_t  = torch.from_numpy((mask_arr > 0).astype(np.int64)).long()
        dmg_t  = torch.from_numpy(mask_arr.astype(np.int64)).long()

        return pre_t, post_t, loc_t, dmg_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--config", type=str, required=True, help="Path to .yaml config")
    parser.add_argument("--data_dir", type=str, default=r"H:\KhoaLuan\data\ida-BD\split\test")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = IdaDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    print(f"Total images: {len(dataset)}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if 'model' in cfg and 'encoder_weights' in cfg['model']:
        cfg['model']['encoder_weights'] = None

    print(f"Creating model from {args.config}...")
    model = create_model(cfg).to(device)

    print(f"Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    print("\n[TENT] Wrapping model with Test-Time Entropy Minimization...")
    tent_model = TENT(model, lr=1e-3, steps=1)

    accumulator = MetricAccumulator()

    print("\n[TENT] Evaluating on ida-BD with online adaptation...")
    for pre_imgs, post_imgs, loc_targets, dmg_targets in tqdm(loader, desc="Evaluating TENT"):
        pre_imgs = pre_imgs.to(device)
        post_imgs = post_imgs.to(device)
        loc_targets = loc_targets.to(device)
        dmg_targets = dmg_targets.to(device)

        with torch.amp.autocast(device.type):
            out_loc, out_dmg = tent_model(pre_imgs, post_imgs)

        accumulator.update(out_loc, out_dmg, loc_targets, dmg_targets)

    metrics = accumulator.compute()
    
    print("\nEVALUATION RESULTS ON ida-BD (TENT)")
    print(f"xView2 Score     : {metrics['xview2_score']:.4f}")
    print(f"F1 Localization  : {metrics['f1_loc']:.4f}")
    print(f"F1 Damage (macro): {metrics['f1_dmg_macro']:.4f}")
    for k in [1, 2, 3, 4]:
        class_name = DAMAGE_CLASS_NAMES[k]
        print(f"  F1 {class_name:<14}: {metrics[f'f1_{class_name}']:.4f}")

if __name__ == "__main__":
    main()
