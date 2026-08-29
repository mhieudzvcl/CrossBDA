"""
dataset.py - xBD DataLoader for Siamese U-Net
Labels: 0=background, 1=no-damage, 2=minor-damage, 3=major-damage, 4=destroyed
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw
from shapely import wkt
from shapely.geometry import mapping
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DAMAGE_LABEL_MAP = {
    'no-damage':     1,
    'minor-damage':  2,
    'major-damage':  3,
    'destroyed':     4,
    'un-classified': 0,
}
NUM_CLASSES = 5   # 0=background, 1-4=damage levels
IMG_SIZE    = 1024


def json_to_mask(label_path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Read an xBD JSON label file and rasterise building polygons to a mask (H, W) with values 0..4."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    with open(label_path, 'r') as f:
        data = json.load(f)

    features = data.get('features', {})
    xy_features = features.get('xy', features.get('lng_lat', []))

    for feat in xy_features:
        props   = feat.get('properties', {})
        subtype = props.get('subtype', 'un-classified')
        label   = DAMAGE_LABEL_MAP.get(subtype, 0)
        if label == 0:
            continue
        wkt_str = feat.get('wkt', '')
        if not wkt_str:
            continue
        try:
            geom   = wkt.loads(wkt_str)
            coords = list(mapping(geom)['coordinates'][0])
            poly   = [(float(x), float(y)) for x, y in coords]
            if len(poly) < 3:
                continue
            pil_mask = Image.fromarray(mask)
            draw = ImageDraw.Draw(pil_mask)
            draw.polygon(poly, fill=int(label))
            mask = np.array(pil_mask)
        except Exception:
            continue
    return mask


def get_train_transforms(img_size: int = 512):
    return A.Compose([
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ], p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'image2': 'image', 'mask2': 'mask'})


def get_val_transforms(img_size: int = 512):
    return A.Compose([
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'image2': 'image', 'mask2': 'mask'})


class XBDDataset(Dataset):
    """
    Returns:
        pre_img  : Tensor (3, H, W) float32
        post_img : Tensor (3, H, W) float32
        loc_mask : Tensor (H, W)    int64  -- building localization (0/1)
        dmg_mask : Tensor (H, W)    int64  -- damage class (0..4)
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        transform=None,
        use_target_png: bool = True,
        split: str = 'train',
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.use_target_png = use_target_png
        self.split = split

        all_files = os.listdir(image_dir)
        pre_files = sorted([f for f in all_files if f.endswith('_pre_disaster.png')])

        self.samples = []
        for pre_name in pre_files:
            base = pre_name.replace('_pre_disaster.png', '')
            post_name   = base + '_post_disaster.png'
            pre_label   = base + '_pre_disaster.json'
            post_label  = base + '_post_disaster.json'

            pre_img_path  = os.path.join(image_dir, pre_name)
            post_img_path = os.path.join(image_dir, post_name)
            post_lbl_path = os.path.join(label_dir, post_label)

            if os.path.exists(post_img_path) and os.path.exists(post_lbl_path):
                self.samples.append({
                    'pre_img':  pre_img_path,
                    'post_img': post_img_path,
                    'pre_lbl':  os.path.join(label_dir, pre_label),
                    'post_lbl': post_lbl_path,
                    'base':     base,
                })

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert('RGB'), dtype=np.uint8)

    def _load_mask(self, sample: dict) -> tuple:
        """Returns (loc_mask, dmg_mask) as numpy arrays (H, W) uint8."""
        dmg_mask = json_to_mask(sample['post_lbl'])

        pre_lbl = sample['pre_lbl']
        if os.path.exists(pre_lbl):
            with open(pre_lbl, 'r') as f:
                data = json.load(f)
            feats    = data.get('features', {})
            xy_feats = feats.get('xy', feats.get('lng_lat', []))
            loc_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for feat in xy_feats:
                wkt_str = feat.get('wkt', '')
                if not wkt_str:
                    continue
                try:
                    geom   = wkt.loads(wkt_str)
                    coords = list(mapping(geom)['coordinates'][0])
                    poly   = [(float(x), float(y)) for x, y in coords]
                    if len(poly) < 3:
                        continue
                    pil_mask = Image.fromarray(loc_mask)
                    draw = ImageDraw.Draw(pil_mask)
                    draw.polygon(poly, fill=1)
                    loc_mask = np.array(pil_mask)
                except Exception:
                    continue
        else:
            loc_mask = (dmg_mask > 0).astype(np.uint8)

        return loc_mask, dmg_mask

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        pre_img  = self._load_image(sample['pre_img'])
        post_img = self._load_image(sample['post_img'])
        loc_mask, dmg_mask = self._load_mask(sample)

        if self.transform:
            augmented = self.transform(
                image=pre_img,
                image2=post_img,
                mask=loc_mask,
                mask2=dmg_mask,
            )
            pre_img  = augmented['image']
            post_img = augmented['image2']
            loc_mask = augmented['mask'].long()
            dmg_mask = augmented['mask2'].long()
        else:
            pre_img  = torch.from_numpy(pre_img.transpose(2, 0, 1)).float() / 255.0
            post_img = torch.from_numpy(post_img.transpose(2, 0, 1)).float() / 255.0
            loc_mask = torch.from_numpy(loc_mask).long()
            dmg_mask = torch.from_numpy(dmg_mask).long()

        return {
            'pre_img':  pre_img,
            'post_img': post_img,
            'loc_mask': loc_mask,
            'dmg_mask': dmg_mask,
            'name':     sample['base'],
        }