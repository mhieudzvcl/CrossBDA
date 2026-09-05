"""
generate_pseudo_labels.py - Generate pseudo labels for Self-Training
"""
import os
import sys
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, r"H:\KhoaLuan")
from src.model import SiameseUNet

def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SiameseUNet(encoder_name="resnet34", encoder_weights=None).to(device)
    ckpt_path = r"H:\KhoaLuan\experiments\fda_results\fda_best_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])
    model.eval()

    img_dir = r"H:\KhoaLuan\data\ida-BD\PRJ-3563"
    out_dir = r"H:\KhoaLuan\data\ida-BD\pseudo_labels"
    os.makedirs(out_dir, exist_ok=True)

    # Search recursively for pre_disaster images
    pre_images = []
    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith("_pre_disaster.png"):
                pre_images.append(os.path.join(root, file))
                
    print(f"Found {len(pre_images)} images to pseudo-label.")

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    with torch.no_grad():
        for pre_path in tqdm(pre_images, desc="Generating"):
            post_path = pre_path.replace("_pre_disaster.png", "_post_disaster.png")
            if not os.path.exists(post_path):
                continue
                
            pre_img  = Image.open(pre_path).convert("RGB").resize((512, 512), Image.BILINEAR)
            post_img = Image.open(post_path).convert("RGB").resize((512, 512), Image.BILINEAR)

            pre_arr  = np.array(pre_img,  dtype=np.float32) / 255.0
            post_arr = np.array(post_img, dtype=np.float32) / 255.0

            pre_arr  = (pre_arr  - mean) / std
            post_arr = (post_arr - mean) / std

            pre_t  = torch.from_numpy(pre_arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            post_t = torch.from_numpy(post_arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

            with torch.amp.autocast(device.type):
                loc_pred, dmg_pred = model(pre_t, post_t)

            loc_mask = torch.argmax(loc_pred, dim=1).squeeze(0).cpu().numpy()
            dmg_mask = torch.argmax(dmg_pred, dim=1).squeeze(0).cpu().numpy()

            final_mask = np.zeros_like(dmg_mask, dtype=np.uint8)
            final_mask[loc_mask > 0] = dmg_mask[loc_mask > 0]
            
            # If damage mask predicted background (0) but loc is > 0, default to no-damage (1)
            final_mask[(loc_mask > 0) & (dmg_mask == 0)] = 1

            # Save the pseudo label using the target format (without pre/post suffix, just the basename of the target mask)
            # The test split uses filename like "Louisiana-East_000000_target.png"
            # Let's extract prefix
            prefix = os.path.basename(pre_path).replace("_pre_disaster.png", "")
            out_path = os.path.join(out_dir, prefix + "_target.png")
            Image.fromarray(final_mask).resize((1024, 1024), Image.NEAREST).save(out_path)

    print(f"Pseudo-labels saved to {out_dir}")

if __name__ == "__main__":
    generate()