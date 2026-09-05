import os
import glob
from PIL import Image

img_dir = r"H:\KhoaLuan\data\ida-BD\PRJ-3563"
mask_dir = r"H:\KhoaLuan\data\ida-BD\pseudo_labels"
pre_images = []
for root, _, files in os.walk(img_dir):
    for file in files:
        if file.endswith("_pre_disaster.png"):
            pre_images.append(os.path.join(root, file))

found = 0
for pre_path in pre_images:
    prefix = os.path.basename(pre_path).replace("_pre_disaster.png", "")
    mask_path = os.path.join(mask_dir, prefix + "_target.png")
    if os.path.exists(mask_path):
        found += 1
print(f"Matched {found} out of {len(pre_images)} masks.")