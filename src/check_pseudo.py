import glob
from PIL import Image
import numpy as np

files = glob.glob('H:/KhoaLuan/data/ida-BD/pseudo_labels/*.png')
total = 0
non_zero = 0
for f in files:
    m = np.array(Image.open(f))
    total += m.size
    non_zero += np.sum(m > 0)
print(f"Foreground pixels: {non_zero/total*100:.2f}%")