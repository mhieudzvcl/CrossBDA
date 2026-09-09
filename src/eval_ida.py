"""
eval_ida.py - Zero-shot evaluation of the xBD-trained model on ida-BD dataset
"""
import os, sys, glob, yaml, warnings
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, r"H:\KhoaLuan")
from src.models.factory import create_model
from src.metrics import MetricAccumulator, DAMAGE_CLASS_NAMES


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


def evaluate():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, default=r"H:\KhoaLuan\data\ida-BD\split\test")
    parser.add_argument("--config",     type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load config and create correct model architecture
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Disable downloading pretrained weights (we load from checkpoint)
    if 'model' in cfg and 'encoder_weights' in cfg['model']:
        cfg['model']['encoder_weights'] = None

    model = create_model(cfg).to(device)

    print('Loading weights...')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    # Strip 'module.' prefix from DataParallel checkpoints
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    if 'epoch' in ckpt:
        print(f'Loaded from epoch {ckpt["epoch"]} (train score: {ckpt.get("score", 0):.4f})')

    dataset = IdaDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    print(f'Total images: {len(dataset)}')

    accumulator = MetricAccumulator()
    with torch.no_grad():
        for pre, post, loc_true, dmg_true in tqdm(loader, desc='Evaluating ida-BD'):
            pre, post = pre.to(device), post.to(device)
            loc_true, dmg_true = loc_true.to(device), dmg_true.to(device)
            with torch.amp.autocast(device.type):
                loc_pred, dmg_pred = model(pre, post)
            accumulator.update(loc_pred, dmg_pred, loc_true, dmg_true)

    results = accumulator.compute()

    print(f'\nEVALUATION RESULTS ON ida-BD (Zero-shot)')
    print(f'xView2 Score     : {results["xview2_score"]:.4f}')
    print(f'F1 Localization  : {results["f1_loc"]:.4f}')
    print(f'F1 Damage (macro): {results["f1_dmg_macro"]:.4f}')
    for name in DAMAGE_CLASS_NAMES[1:]:
        key = f'f1_{name}'
        print(f'F1 {name:<16}: {results.get(key, 0):.4f}')


if __name__ == "__main__":
    evaluate()
