"""
eval_s12.py - Zero-shot evaluation of the xBD-trained model on S12 dataset (Sentinel-2 TCI)
"""
import os, sys, glob, warnings
import numpy as np
import torch
import cv2
import tifffile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model   import SiameseUNet
from src.metrics import MetricAccumulator, DAMAGE_CLASS_NAMES

class S12Dataset(Dataset):
    def __init__(self, s12_dir, xbd_dir):
        """
        s12_dir: Thu muc chua s2_tci, e.g., G:/My Drive/KhoaLuan_Data/data/xBD-S12/s2_tci
        xbd_dir: Thu muc xBD goc de lay mask, e.g., G:/My Drive/KhoaLuan_Data/data/xBD
        """
        self.s12_dir = s12_dir
        self.xbd_dir = xbd_dir
        self.pre_images = sorted(glob.glob(os.path.join(s12_dir, "*_pre_disaster_s2_tci.tif")))
        
        # Build a lookup table for masks from all xBD splits
        self.mask_lookup = {}
        for split in ['train', 'test', 'hold', 'tier3']:
            mask_dir = os.path.join(xbd_dir, split, "targets")
            if os.path.exists(mask_dir):
                for f in os.listdir(mask_dir):
                    if f.endswith('_post_disaster_target.png'):
                        self.mask_lookup[f] = os.path.join(mask_dir, f)
                        
        print(f"Found {len(self.pre_images)} Sentinel-2 image pairs.")
        print(f"Found {len(self.mask_lookup)} xBD masks in {xbd_dir}.")

        # Filter only those that have a corresponding mask
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
        
        # Read TIF (128x128x3 RGB uint8)
        pre_img = tifffile.imread(pair['pre'])
        post_img = tifffile.imread(pair['post'])
        
        # Resize tu 128x128 len 1024x1024 bang cv2 (Bilinear)
        pre_img_1024 = cv2.resize(pre_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        post_img_1024 = cv2.resize(post_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        
        # Chuyen ve Tensor, chuan hoa [0, 1]
        pre_tensor = torch.from_numpy(pre_img_1024.transpose(2, 0, 1)).float() / 255.0
        post_tensor = torch.from_numpy(post_img_1024.transpose(2, 0, 1)).float() / 255.0
        
        # Load mask goc tu xBD (1024x1024)
        mask = Image.open(pair['mask'])
        mask = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask)
        
        return pre_tensor, post_tensor, mask_tensor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir_xbd = r"H:\KhoaLuan\data\xBD"
    data_dir_s12 = r"H:\KhoaLuan\data\xBD-S12\s2_tci"
    ckpt_path    = r"H:\KhoaLuan\experiments\baseline_resnet34\checkpoints\best_model.pth"

    # Fallback to check if scratch path has it (for testing)
    if not os.path.exists(data_dir_s12):
        data_dir_s12 = r"H:\KhoaLuan\scratch\xbd_s12\s2_tci"

    dataset = S12Dataset(s12_dir=data_dir_s12, xbd_dir=data_dir_xbd)
    if len(dataset) == 0:
        print("Loi: Khong tim thay du lieu de danh gia. Kiem tra lai duong dan data_dir_s12 va data_dir_xbd.")
        return
        
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    print("Loading weights...")
    model = SiameseUNet(encoder_name='resnet34').to(device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'], strict=False)
    else:
        print(f"Khong tim thay checkpoint tai {ckpt_path}!")
        return
    model.eval()

    accumulator = MetricAccumulator()

    print(f"Total images to evaluate: {len(dataset)}")
    with torch.no_grad():
        for pre_imgs, post_imgs, targets in tqdm(loader, desc="Evaluating xBD-S12"):
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            targets = targets.to(device)

            out_loc, out_dmg = model(pre_imgs, post_imgs)

            loc_targets = (targets > 0).long()
            dmg_targets = targets.long()
            accumulator.update(out_loc, out_dmg, loc_targets, dmg_targets)

    metrics = accumulator.compute()
    
    print("\nEVALUATION RESULTS ON xBD-S12 (Zero-shot)")
    print(f"xView2 Score     : {metrics['xview2_score']:.4f}")
    print(f"F1 Localization  : {metrics['f1_loc']:.4f}")
    print(f"F1 Damage (macro): {metrics['f1_dmg_macro']:.4f}")
    
    for k in [1, 2, 3, 4]:
        class_name = DAMAGE_CLASS_NAMES[k]
        f1_val = metrics[f'f1_{class_name}']
        print(f"  F1 {class_name:<14}: {f1_val:.4f}")

if __name__ == "__main__":
    main()