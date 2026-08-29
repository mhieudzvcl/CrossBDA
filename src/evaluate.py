"""
evaluate.py - Evaluate trained model on xBD test or holdout split
Usage:
    python src/evaluate.py --checkpoint experiments/baseline_resnet34/checkpoints/best_model.pth --split test
"""
import os, sys, yaml, argparse, warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import XBDDataset, get_val_transforms
from src.model   import SiameseUNet
from src.metrics import MetricAccumulator, DAMAGE_CLASS_NAMES


def evaluate(cfg_path, ckpt_path, split='test'):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Split: {split}')

    data_root = Path(cfg['data']['root'])
    img_dir   = data_root / split / 'images'
    lbl_dir   = data_root / split / 'labels'

    img_size = cfg['data']['img_size']
    dataset  = XBDDataset(str(img_dir), str(lbl_dir), transform=get_val_transforms(img_size))
    loader   = DataLoader(dataset, batch_size=4, shuffle=False,
                          num_workers=cfg['data']['num_workers'], pin_memory=True)

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SiameseUNet(
        encoder_name=cfg['model']['encoder'],
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded from epoch {ckpt["epoch"]} (train score: {ckpt["score"]:.4f})')

    accumulator = MetricAccumulator()
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            pre    = batch['pre_img'].to(device)
            post   = batch['post_img'].to(device)
            loc_gt = batch['loc_mask'].to(device)
            dmg_gt = batch['dmg_mask'].to(device)
            loc_logits, dmg_logits = model(pre, post)
            accumulator.update(loc_logits, dmg_logits, loc_gt, dmg_gt)

    metrics = accumulator.compute()

    print(f'\nEVALUATION RESULTS ({split.upper()})')
    print(f'xView2 Score     : {metrics["xview2_score"]:.4f}')
    print(f'F1 Localization  : {metrics["f1_loc"]:.4f}')
    print(f'F1 Damage (macro): {metrics["f1_dmg_macro"]:.4f}')
    for name in DAMAGE_CLASS_NAMES[1:]:
        key = f'f1_{name}'
        print(f'F1 {name:<16}: {metrics.get(key, 0):.4f}')

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/baseline.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split',      default='test', choices=['test', 'hold'])
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.split)