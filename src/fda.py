"""
fda.py - Fourier Domain Adaptation for Building Damage Assessment
Exp 3.1: Unsupervised domain alignment between xBD (source) and ida-BD (target)

Algorithm:
  1. Apply FFT2D to source image (xBD) and target image (ida-BD)
  2. Extract Amplitude of target (encodes color/style)
  3. Replace the low-frequency Amplitude of source with target's Amplitude
  4. Apply inverse FFT -> "style-transferred" image that looks like ida-BD
     but retains xBD structure/labels

Usage modes:
  (A) Visualization:  python src/fda.py --mode vis
  (B) Evaluate model on FDA-transformed ida-BD: python src/fda.py --mode eval
"""

import os, sys, glob, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model   import SiameseUNet
from src.metrics import MetricAccumulator, DAMAGE_CLASS_NAMES

# ============================================================
# CORE FDA TRANSFORM
# ============================================================

def fda_transform(src_img: np.ndarray, tgt_img: np.ndarray, beta: float = 0.01) -> np.ndarray:
    """
    Apply Fourier Domain Adaptation to a single image pair.

    Args:
        src_img : Source image (H, W, 3) float32 [0, 255]
        tgt_img : Target image (H, W, 3) float32 [0, 255]
        beta    : Size of the low-frequency region to swap (0.01 ~ 0.1).
                  Smaller = more subtle style transfer.

    Returns:
        Adapted source image (H, W, 3) float32 [0, 255]
    """
    assert src_img.shape == tgt_img.shape, "Source and target must have same shape"
    h, w = src_img.shape[:2]

    # Size of the frequency band to replace
    b = int(np.floor(min(h, w) * beta))
    if b == 0:
        return src_img.copy()

    result = np.zeros_like(src_img)

    for c in range(3):
        # FFT on each channel
        src_fft = np.fft.fft2(src_img[:, :, c])
        tgt_fft = np.fft.fft2(tgt_img[:, :, c])

        # Shift zero frequency to center
        src_fft_shift = np.fft.fftshift(src_fft)
        tgt_fft_shift = np.fft.fftshift(tgt_fft)

        # Extract amplitude and phase
        src_amp, src_pha = np.abs(src_fft_shift), np.angle(src_fft_shift)
        tgt_amp          = np.abs(tgt_fft_shift)

        # Swap the low-frequency amplitude region (center square [b x b])
        cy, cx = h // 2, w // 2
        src_amp_new = src_amp.copy()
        src_amp_new[cy - b: cy + b, cx - b: cx + b] = \
            tgt_amp[cy - b: cy + b, cx - b: cx + b]

        # Reconstruct: new_amplitude * e^(i * src_phase)
        src_fft_new  = src_amp_new * np.exp(1j * src_pha)
        src_fft_back = np.fft.ifftshift(src_fft_new)
        src_img_back = np.fft.ifft2(src_fft_back).real

        result[:, :, c] = np.clip(src_img_back, 0, 255)

    return result.astype(np.float32)


# ============================================================
# DATASET WITH FDA AUGMENTATION
# ============================================================

class IdaDatasetFDA(Dataset):
    """
    Loads ida-BD images and applies FDA transform using random xBD images as style reference.
    The model will see ida-BD images "colored" like xBD training data.

    Wait -- Actually for evaluation we want to transform the xBD-style idas look,
    meaning we pass in the actual ida-BD images AS IS and evaluate.
    For training augmentation, we transform xBD images to look like ida-BD.

    For Exp 3.1 eval: We use ida-BD images directly (no transform needed).
    This script shows: what happens if we TRAIN with FDA-augmented data.
    """
    def __init__(self, ida_dir: str, xbd_img_dir: str = None, beta: float = 0.01,
                 apply_fda: bool = False):
        self.img_dir   = os.path.join(ida_dir, "images")
        self.mask_dir  = os.path.join(ida_dir, "masks")
        self.apply_fda = apply_fda
        self.beta      = beta

        self.pre_images = sorted(glob.glob(os.path.join(self.img_dir, "*_pre_disaster.png")))

        # Collect xBD reference images for style transfer
        self.xbd_refs = []
        if xbd_img_dir and apply_fda:
            for split in ["train"]:
                path = os.path.join(xbd_img_dir, split, "images")
                if os.path.exists(path):
                    imgs = glob.glob(os.path.join(path, "*_post_disaster.png"))
                    self.xbd_refs.extend(imgs)
            print(f"FDA: loaded {len(self.xbd_refs)} xBD reference images for style transfer")

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_path  = self.pre_images[idx]
        post_path = pre_path.replace("_pre_", "_post_")
        mask_path = os.path.join(self.mask_dir, os.path.basename(post_path))

        pre_img  = np.array(Image.open(pre_path).convert("RGB").resize((512, 512), Image.BILINEAR), dtype=np.float32)
        post_img = np.array(Image.open(post_path).convert("RGB").resize((512, 512), Image.BILINEAR), dtype=np.float32)
        mask     = np.array(Image.open(mask_path).resize((512, 512), Image.NEAREST))

        if self.apply_fda and self.xbd_refs:
            # Pick a random xBD image as the "target style" to imprint onto ida images
            ref_path = random.choice(self.xbd_refs)
            ref_img  = np.array(Image.open(ref_path).convert("RGB").resize((512, 512), Image.BILINEAR), dtype=np.float32)
            pre_img  = fda_transform(pre_img,  ref_img, self.beta)
            post_img = fda_transform(post_img, ref_img, self.beta)

        # Normalize to [0, 1] then ImageNet normalize
        pre_arr  = pre_img  / 255.0
        post_arr = post_img / 255.0

        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        pre_arr  = (pre_arr  - mean) / std
        post_arr = (post_arr - mean) / std

        pre_t  = torch.from_numpy(pre_arr.transpose(2, 0, 1)).float()
        post_t = torch.from_numpy(post_arr.transpose(2, 0, 1)).float()
        loc_t  = torch.from_numpy((mask > 0).astype(np.int64)).long()
        dmg_t  = torch.from_numpy(mask.astype(np.int64)).long()

        return pre_t, post_t, loc_t, dmg_t


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_fda(ida_dir: str, xbd_dir: str, out_dir: str, n_samples: int = 4,
                  betas: list = [0.01, 0.05, 0.1]):
    """
    Save side-by-side comparison images:
    [Original ida-BD] | [FDA beta=0.01] | [FDA beta=0.05] | [FDA beta=0.1]
    """
    os.makedirs(out_dir, exist_ok=True)

    ida_imgs = sorted(glob.glob(os.path.join(ida_dir, "images", "*_post_disaster.png")))[:n_samples]

    xbd_refs = []
    for split in ["train"]:
        p = os.path.join(xbd_dir, split, "images")
        if os.path.exists(p):
            xbd_refs.extend(glob.glob(os.path.join(p, "*_post_disaster.png")))

    if not xbd_refs:
        print(f"ERROR: No xBD reference images found in {xbd_dir}")
        return

    print(f"Generating FDA visualization for {len(ida_imgs)} ida-BD images...")
    for i, ida_path in enumerate(ida_imgs):
        ida_img = np.array(Image.open(ida_path).convert("RGB").resize((512, 512)), dtype=np.float32)
        ref_img = np.array(Image.open(random.choice(xbd_refs)).convert("RGB").resize((512, 512)), dtype=np.float32)

        cols = [ida_img]
        labels = ["ida-BD (original)"]
        for b in betas:
            adapted = fda_transform(ida_img, ref_img, beta=b)
            cols.append(adapted)
            labels.append(f"FDA beta={b}")

        # Stack side by side
        row = np.hstack([np.clip(c, 0, 255).astype(np.uint8) for c in cols])

        # Add text labels
        label_row = np.ones((40, row.shape[1], 3), dtype=np.uint8) * 240
        x_step = row.shape[1] // len(cols)
        for j, lbl in enumerate(labels):
            cv2.putText(label_row, lbl, (j * x_step + 5, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)

        final = np.vstack([label_row, row])
        out_path = os.path.join(out_dir, f"fda_sample_{i:02d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {out_path}")

    print(f"\nVisualization complete! Open the folder: {out_dir}")


# ============================================================
# EVALUATION WITH FDA-TRANSFORMED INPUTS
# ============================================================

def evaluate_with_fda(ida_dir: str, xbd_dir: str, ckpt_path: str, beta: float = 0.05):
    """
    Evaluate the model on ida-BD images that have been FDA-transformed
    to "look like" xBD training data.
    This tests if FDA alignment improves zero-shot performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | FDA beta: {beta}")

    model = SiameseUNet(encoder_name="resnet34", encoder_weights=None).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Weights loaded.")

    # We transform ida-BD images to look like xBD
    dataset = IdaDatasetFDA(ida_dir=ida_dir, xbd_img_dir=xbd_dir, beta=beta, apply_fda=True)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Total images: {len(dataset)}")

    accumulator = MetricAccumulator()
    with torch.no_grad():
        for pre, post, loc_t, dmg_t in tqdm(loader, desc=f"Evaluating FDA (beta={beta})"):
            pre, post = pre.to(device), post.to(device)
            loc_t, dmg_t = loc_t.to(device), dmg_t.to(device)
            out_loc, out_dmg = model(pre, post)
            accumulator.update(out_loc, out_dmg, loc_t, dmg_t)

    metrics = accumulator.compute()
    print(f"\nEVALUATION RESULTS ON ida-BD (FDA Zero-shot, beta={beta})")
    print(f"xView2 Score     : {metrics['xview2_score']:.4f}")
    print(f"F1 Localization  : {metrics['f1_loc']:.4f}")
    print(f"F1 Damage (macro): {metrics['f1_dmg_macro']:.4f}")
    for k in [1, 2, 3, 4]:
        name = DAMAGE_CLASS_NAMES[k]
        print(f"  F1 {name:<14}: {metrics[f'f1_{name}']:.4f}")
    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fourier Domain Adaptation")
    parser.add_argument("--mode",     type=str, default="vis",
                        choices=["vis", "eval"],
                        help="vis=visualization, eval=run evaluation")
    parser.add_argument("--ida_dir",  type=str,
                        default=r"H:\KhoaLuan\data\ida-BD\PRJ-3563\Project--ida-bd-pre-and-post-disaster-high-resolution-satellite-imagery-for-building-damage-assessment-from-hurricane-ida\data")
    parser.add_argument("--xbd_dir",  type=str,
                        default=r"H:\KhoaLuan\data\xBD")
    parser.add_argument("--ckpt",     type=str,
                        default=r"H:\KhoaLuan\experiments\baseline_resnet34\checkpoints\best_model.pth")
    parser.add_argument("--out_dir",  type=str,
                        default=r"H:\KhoaLuan\experiments\fda_results\vis")
    parser.add_argument("--beta",     type=float, default=0.05,
                        help="FDA low-frequency swap ratio (0.01 to 0.1)")
    parser.add_argument("--n_vis",    type=int,   default=5,
                        help="Number of visualization samples")
    args = parser.parse_args()

    if args.mode == "vis":
        visualize_fda(
            ida_dir=args.ida_dir,
            xbd_dir=args.xbd_dir,
            out_dir=args.out_dir,
            n_samples=args.n_vis,
            betas=[0.01, 0.05, 0.10]
        )
    elif args.mode == "eval":
        evaluate_with_fda(
            ida_dir=args.ida_dir,
            xbd_dir=args.xbd_dir,
            ckpt_path=args.ckpt,
            beta=args.beta
        )