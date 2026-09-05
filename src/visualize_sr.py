import os, glob, cv2, tifffile
from cv2 import dnn_superres
import shutil

s12_dir = r"H:\KhoaLuan\data\xBD-S12\s2_tci"
out_dir = r"H:\KhoaLuan\experiments\visualize_sr"
os.makedirs(os.path.join(out_dir, "original_128"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "cv2_resize_1024"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "lapsrn_1024"), exist_ok=True)

# Load SR model
sr = dnn_superres.DnnSuperResImpl_create()
path = r"H:\KhoaLuan\experiments\LapSRN_x8.pb"
sr.readModel(path)
sr.setModel("lapsrn", 8)
print("LapSRN x8 loaded.")

images = sorted(glob.glob(os.path.join(s12_dir, "*_pre_disaster_s2_tci.tif")))[:5]

for img_path in images:
    base_name = os.path.basename(img_path).replace('.tif', '.png')
    
    # Read TIF (128x128)
    img_rgb = tifffile.imread(img_path)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Original (save as PNG)
    cv2.imwrite(os.path.join(out_dir, "original_128", base_name), img_bgr)
    
    # 2. cv2.resize
    resized = cv2.resize(img_bgr, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, "cv2_resize_1024", base_name), resized)
    
    # 3. LapSRN
    upscaled = sr.upsample(img_bgr)
    cv2.imwrite(os.path.join(out_dir, "lapsrn_1024", base_name), upscaled)
    
    print(f"Processed {base_name}")

print(f"Saved visualization images to {out_dir}")