"""
eval_ida_patch_fda.py - Evaluate Scale-MAE with Patch-Level FDA at Inference Time.

Novel contribution: Apply Patch-Level FDA (patch_size=16 matching ViT patch size)
to transform Ida-BD images toward xBD domain style at test time.
No retraining needed. Works with any pre-trained Scale-MAE checkpoint.

Usage:
    python src/eval_ida_patch_fda.py \
        --checkpoint "H:\KhoaLuan\best_model_ScaleMAE_0.7603.pth" \
        --config configs/scalemae.yaml \
        --xbd_style_dir "H:\KhoaLuan\data\xBD\test\images" \
        --alpha 0.5 --beta 0.01
"""
import os, sys, glob, yaml, warnings
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import random

warnings.filterwarnings('ignore')
sys.path.insert(0, r"H:\KhoaLuan")
from src.models.factory import create_model
from src.metrics import MetricAccumulator, DAMAGE_CLASS_NAMES
from src.fda_patch import PatchFDA, global_fda


class IdaDatasetWithPatchFDA(Dataset):
    """
    Ida-BD dataset with optional Patch-Level FDA applied at load time.
    Style reference images are sampled from xBD to transfer color/texture domain.
    """
    def __init__(self, data_dir, style_images: list, patch_size=16, beta=0.01, alpha=0.5, mode='patch'):
        """
        Args:
            data_dir: Path to ida-BD split/test
            style_images: List of numpy [H,W,3] images from xBD (style references)
            patch_size: ViT patch size (16 for Scale-MAE)
            beta: FDA low-frequency radius ratio
            alpha: Amplitude blending (0=no change, 1=full FDA)
            mode: 'patch' (Patch-Level FDA), 'global' (Standard FDA), 'none' (baseline)
        """
        self.img_dir  = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.pre_images = sorted(glob.glob(os.path.join(self.img_dir, "*_pre_disaster.png")))
        self.mode = mode

        if mode == 'patch':
            self.fda = PatchFDA(patch_size=patch_size, beta=beta, alpha=alpha, target_images=style_images)
        self.style_images = style_images
        self.beta = beta
        self.alpha = alpha

        print(f"Ida-BD: {len(self.pre_images)} images | FDA mode: {mode} | alpha={alpha}, beta={beta}")

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_path  = self.pre_images[idx]
        post_path = pre_path.replace("_pre_", "_post_")
        mask_path = os.path.join(self.mask_dir, os.path.basename(post_path))

        pre_img  = np.array(Image.open(pre_path).convert("RGB").resize((512, 512), Image.BILINEAR), dtype=np.uint8)
        post_img = np.array(Image.open(post_path).convert("RGB").resize((512, 512), Image.BILINEAR), dtype=np.uint8)
        mask_arr = np.array(Image.open(mask_path).resize((512, 512), Image.NEAREST))

        # Fixed: ONE shared reference per pair (pre + post use SAME style image)
        if self.mode != 'none' and self.style_images:
            tgt = self.style_images[random.randint(0, len(self.style_images) - 1)]
            if self.mode == 'patch':
                pre_img  = self.fda(pre_img, tgt)
                post_img = self.fda(post_img, tgt)
            elif self.mode == 'global':
                pre_img  = global_fda(pre_img, tgt, beta=self.beta, alpha=self.alpha)
                post_img = global_fda(post_img, tgt, beta=self.beta, alpha=self.alpha)

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        pre_arr  = pre_img.astype(np.float32) / 255.0
        post_arr = post_img.astype(np.float32) / 255.0
        pre_arr  = (pre_arr  - mean) / std
        post_arr = (post_arr - mean) / std

        pre_t  = torch.from_numpy(pre_arr.transpose(2, 0, 1)).float()
        post_t = torch.from_numpy(post_arr.transpose(2, 0, 1)).float()
        loc_t  = torch.from_numpy((mask_arr > 0).astype(np.int64)).long()
        dmg_t  = torch.from_numpy(mask_arr.astype(np.int64)).long()

        return pre_t, post_t, loc_t, dmg_t


def load_style_images(xbd_style_dir: str, n_samples: int = 50) -> list:
    """Load a random sample of xBD images to use as FDA style references."""
    all_imgs = sorted(glob.glob(os.path.join(xbd_style_dir, "*_pre_disaster.png")))
    if not all_imgs:
        all_imgs = sorted(glob.glob(os.path.join(xbd_style_dir, "*.png")))[:200]
    
    random.seed(42)
    selected = random.sample(all_imgs, min(n_samples, len(all_imgs)))
    style_imgs = []
    for path in selected:
        img = np.array(Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR), dtype=np.uint8)
        style_imgs.append(img)
    
    print(f"Loaded {len(style_imgs)} xBD style reference images from {xbd_style_dir}")
    return style_imgs


def run_eval(model, loader, device, label: str) -> dict:
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for pre, post, loc_true, dmg_true in tqdm(loader, desc=f"Evaluating [{label}]"):
            pre, post = pre.to(device), post.to(device)
            loc_true, dmg_true = loc_true.to(device), dmg_true.to(device)
            with torch.amp.autocast(device.type):
                loc_pred, dmg_pred = model(pre, post)
            accumulator.update(loc_pred, dmg_pred, loc_true, dmg_true)
    return accumulator.compute()


def print_results(results: dict, label: str):
    print(f"\n{'='*55}")
    print(f"  RESULTS: {label}")
    print(f"{'='*55}")
    print(f"  xView2 Score     : {results['xview2_score']:.4f}")
    print(f"  F1 Localization  : {results['f1_loc']:.4f}")
    print(f"  F1 Damage (macro): {results['f1_dmg_macro']:.4f}")
    for name in DAMAGE_CLASS_NAMES[1:]:
        print(f"    F1 {name:<14}: {results.get(f'f1_{name}', 0):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--config",        type=str, default="configs/scalemae.yaml")
    parser.add_argument("--data_dir",      type=str, default=r"H:\KhoaLuan\data\ida-BD\split\test")
    parser.add_argument("--xbd_style_dir", type=str, default=r"H:\KhoaLuan\data\xBD\test\images",
                        help="Directory with xBD images used as FDA style references")
    parser.add_argument("--alpha",  type=float, default=0.5, help="FDA blending factor (0-1)")
    parser.add_argument("--beta",   type=float, default=0.01, help="FDA low-freq radius")
    parser.add_argument("--n_style",type=int,   default=50,   help="Number of style reference images")
    parser.add_argument("--compare", action="store_true",
                        help="If set, run all 3 modes (none/global/patch) for comparison")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if 'model' in cfg and 'encoder_weights' in cfg['model']:
        cfg['model']['encoder_weights'] = None

    model = create_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    if 'epoch' in ckpt:
        print(f"Loaded epoch {ckpt['epoch']} (train score: {ckpt.get('score', 0):.4f})")

    # Load style reference images
    style_images = load_style_images(args.xbd_style_dir, n_samples=args.n_style)

    if args.compare:
        # Run all 3 modes for comparison table
        for mode in ['none', 'global', 'patch']:
            ds = IdaDatasetWithPatchFDA(
                args.data_dir, style_images,
                patch_size=16, beta=args.beta, alpha=args.alpha, mode=mode
            )
            loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
            label = {
                'none': 'Zero-shot (no FDA)',
                'global': f'Global FDA (alpha={args.alpha})',
                'patch': f'Patch-Level FDA p=16 (alpha={args.alpha})'
            }[mode]
            results = run_eval(model, loader, device, label)
            print_results(results, label)
    else:
        # Run Patch-Level FDA only
        ds = IdaDatasetWithPatchFDA(
            args.data_dir, style_images,
            patch_size=16, beta=args.beta, alpha=args.alpha, mode='patch'
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
        results = run_eval(model, loader, device, f"Patch-Level FDA (alpha={args.alpha})")
        print_results(results, f"Patch-Level FDA p=16 alpha={args.alpha}")


if __name__ == "__main__":
    main()
