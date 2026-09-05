import os, sys, glob, warnings
import numpy as np
import torch
import cv2
import tifffile
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from cv2 import dnn_superres

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SiameseUNet
from src.metrics import MetricAccumulator, DAMAGE_CLASS_NAMES

class S12DatasetSR(Dataset):
    def __init__(self, s12_dir, xbd_dir, limit=100):
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
                    if len(self.valid_pairs) >= limit:
                        break
        print(f"Loaded {len(self.valid_pairs)} pairs for SR evaluation.")
        
        # Init SR model
        self.sr = dnn_superres.DnnSuperResImpl_create()
        path = r"H:\KhoaLuan\experiments\LapSRN_x8.pb"
        self.sr.readModel(path)
        self.sr.setModel("lapsrn", 8)
        print("LapSRN x8 model loaded successfully.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        # Read TIF (128x128x3 RGB uint8)
        pre_img = tifffile.imread(pair['pre'])
        post_img = tifffile.imread(pair['post'])
        
        # Convert RGB to BGR for OpenCV
        pre_bgr = cv2.cvtColor(pre_img, cv2.COLOR_RGB2BGR)
        post_bgr = cv2.cvtColor(post_img, cv2.COLOR_RGB2BGR)
        
        # Upscale 128x128 -> 1024x1024 using LapSRN x8
        pre_sr = self.sr.upsample(pre_bgr)
        post_sr = self.sr.upsample(post_bgr)
        
        # Convert back to RGB
        pre_sr = cv2.cvtColor(pre_sr, cv2.COLOR_BGR2RGB)
        post_sr = cv2.cvtColor(post_sr, cv2.COLOR_BGR2RGB)
        
        # To Tensor [0, 1]
        pre_tensor = torch.from_numpy(pre_sr.transpose(2, 0, 1)).float() / 255.0
        post_tensor = torch.from_numpy(post_sr.transpose(2, 0, 1)).float() / 255.0
        
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

    dataset = S12DatasetSR(s12_dir=data_dir_s12, xbd_dir=data_dir_xbd, limit=100)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    print("Loading weights...")
    model = SiameseUNet(encoder_name='resnet34').to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'], strict=False)
    model.eval()

    accumulator = MetricAccumulator()

    with torch.no_grad():
        for pre_imgs, post_imgs, targets in tqdm(loader, desc="Evaluating SR xBD-S12"):
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            targets = targets.to(device)

            out_loc, out_dmg = model(pre_imgs, post_imgs)

            loc_targets = (targets > 0).long()
            dmg_targets = targets.long()
            accumulator.update(out_loc, out_dmg, loc_targets, dmg_targets)

    metrics = accumulator.compute()
    
    print("\nEVALUATION RESULTS ON xBD-S12 (Super-Resolution)")
    print(f"xView2 Score     : {metrics['xview2_score']:.4f}")
    print(f"F1 Localization  : {metrics['f1_loc']:.4f}")
    print(f"F1 Damage (macro): {metrics['f1_dmg_macro']:.4f}")

if __name__ == "__main__":
    main()