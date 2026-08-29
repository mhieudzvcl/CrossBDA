import numpy as np
from PIL import Image
mask_path = r"H:\KhoaLuan\data\ida-BD\PRJ-3563\Project--ida-bd-pre-and-post-disaster-high-resolution-satellite-imagery-for-building-damage-assessment-from-hurricane-ida\data\masks\AOI1-tile_1-3_post_disaster.png"
arr = np.array(Image.open(mask_path))
print("Shape:", arr.shape)
print("Unique values:", np.unique(arr))