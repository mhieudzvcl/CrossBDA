import os
import glob
import random
import shutil

src_dir = r'H:\KhoaLuan\data\ida-BD\PRJ-3563\Project--ida-bd-pre-and-post-disaster-high-resolution-satellite-imagery-for-building-damage-assessment-from-hurricane-ida\data'
dst_dir = r'H:\KhoaLuan\data\ida-BD\split'

random.seed(42) # For reproducibility

pre_images = sorted(glob.glob(os.path.join(src_dir, 'images', '*_pre_disaster.png')))
print(f'Found {len(pre_images)} total image pairs.')

# Shuffle
random.shuffle(pre_images)

# Split
train_10_pre = pre_images[:10]
train_5_pre = train_10_pre[:5]
test_pre = pre_images[10:]

print(f'Train 10: {len(train_10_pre)}')
print(f'Train 5: {len(train_5_pre)}')
print(f'Test: {len(test_pre)}')

def copy_files(pre_list, out_folder):
    out_img = os.path.join(dst_dir, out_folder, 'images')
    out_mask = os.path.join(dst_dir, out_folder, 'masks')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_mask, exist_ok=True)
    
    for pre_path in pre_list:
        post_path = pre_path.replace('_pre_', '_post_')
        mask_path = os.path.join(src_dir, 'masks', os.path.basename(post_path))
        
        shutil.copy2(pre_path, os.path.join(out_img, os.path.basename(pre_path)))
        shutil.copy2(post_path, os.path.join(out_img, os.path.basename(post_path)))
        if os.path.exists(mask_path):
            shutil.copy2(mask_path, os.path.join(out_mask, os.path.basename(mask_path)))

print('Copying files for train_10...')
copy_files(train_10_pre, 'train_10')

print('Copying files for train_5...')
copy_files(train_5_pre, 'train_5')

print('Copying files for test...')
copy_files(test_pre, 'test')

print('Done!')