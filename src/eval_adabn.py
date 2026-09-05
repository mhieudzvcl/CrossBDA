"""
eval_adabn.py - Adaptive Batch Normalization for Test-Time Domain Adaptation
Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation", 2016.

Fix from v1: use momentum=None (cumulative moving average) so that after 1 pass
over all test images, running_mean/var = TRUE mean/var of target domain.
momentum=0.1 (default) would only capture 10% of the target statistics per pass.
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


def prepare_bn_for_adaptation(model):
    """
    Set BN layers to train mode with momentum=None (cumulative moving average).
    This ensures 1 full pass gives the TRUE mean/var of target domain,
    not just 10% (which default momentum=0.1 would give).
    """
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train()
            m.reset_running_stats()
            m.momentum = None   # KEY FIX: cumulative moving average


def adapt_bn(model, loader, device):
    """
    Forward pass over ALL test images to accumulate stable BN statistics.
    No labels, no backprop needed.
    """
    prepare_bn_for_adaptation(model)
    print(f"AdaBN: calibrating BN stats over {len(loader.dataset)} images...")
    with torch.no_grad():
        for pre, post, _, _ in tqdm(loader, desc="  BN Calibration"):
            pre, post = pre.to(device), post.to(device)
            model(pre, post)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, default=r"H:\KhoaLuan\data\ida-BD\split\test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SiameseUNet(encoder_name="resnet34", encoder_weights=None).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print("Weights loaded.")

    dataset = IdaDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    print(f"Total images: {len(dataset)}")

    # Phase 1: BN Adaptation (1 pass, cumulative moving average)
    model = adapt_bn(model, loader, device)

    # Phase 2: Evaluation with adapted stats
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for pre, post, loc_true, dmg_true in tqdm(loader, desc="Evaluating AdaBN"):
            pre, post = pre.to(device), post.to(device)
            loc_true, dmg_true = loc_true.to(device), dmg_true.to(device)
            loc_pred, dmg_pred = model(pre, post)
            accumulator.update(loc_pred, dmg_pred, loc_true, dmg_true)

    results = accumulator.compute()
    print(f"\nEVALUATION RESULTS ON ida-BD (AdaBN - momentum=None)")
    print(f"xView2 Score     : {results['xview2_score']:.4f}")
    print(f"F1 Localization  : {results['f1_loc']:.4f}")
    print(f"F1 Damage (macro): {results['f1_dmg_macro']:.4f}")
    for name in DAMAGE_CLASS_NAMES[1:]:
        print(f"F1 {name:<16}: {results.get('f1_' + name, 0):.4f}")
