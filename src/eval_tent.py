"""
eval_tent.py - Evaluate SiameseUNet with TENT on ida-BD
"""
import os, sys, glob, warnings
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

from model   import SiameseUNet
from metrics import MetricAccumulator, DAMAGE_CLASS_NAMES
from tent    import TENT


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, default=r"H:\KhoaLuan\data\ida-BD\split\test")
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--steps",      type=int,   default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | TENT lr={args.lr} steps={args.steps}")

    base_model = SiameseUNet(encoder_name="resnet34", encoder_weights=None).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_model.load_state_dict(ckpt["model_state"], strict=False)
    print("Weights loaded.")

    tent_model = TENT(base_model, lr=args.lr, steps=args.steps)

    dataset = IdaDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Total images: {len(dataset)}")

    accumulator = MetricAccumulator()
    for pre, post, loc_true, dmg_true in tqdm(loader, desc="Evaluating TENT"):
        pre, post = pre.to(device), post.to(device)
        loc_true, dmg_true = loc_true.to(device), dmg_true.to(device)
        loc_pred, dmg_pred = tent_model(pre, post)
        accumulator.update(loc_pred, dmg_pred, loc_true, dmg_true)

    results = accumulator.compute()
    print(f"\nEVALUATION RESULTS ON ida-BD (TENT lr={args.lr} steps={args.steps})")
    print(f"xView2 Score     : {results['xview2_score']:.4f}")
    print(f"F1 Localization  : {results['f1_loc']:.4f}")
    print(f"F1 Damage (macro): {results['f1_dmg_macro']:.4f}")
    for name in DAMAGE_CLASS_NAMES[1:]:
        print(f"F1 {name:<16}: {results.get('f1_' + name, 0):.4f}")
