"""
train.py - Training loop for Siamese U-Net (xBD)
"""
import os, sys, yaml, argparse, random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import XBDDataset, get_train_transforms, get_val_transforms
from src.model   import SiameseUNet
from src.losses  import CombinedLoss
from src.metrics import MetricAccumulator, AverageMeter


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    pbar = tqdm(loader, desc=f'[Train] Epoch {epoch}', leave=False)
    for batch in pbar:
        pre    = batch['pre_img'].to(device, non_blocking=True)
        post   = batch['post_img'].to(device, non_blocking=True)
        loc_gt = batch['loc_mask'].to(device, non_blocking=True)
        dmg_gt = batch['dmg_mask'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            loc_logits, dmg_logits = model(pre, post)
            loss, _ = criterion(loc_logits, dmg_logits, loc_gt, dmg_gt)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_meter.update(loss.item())
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    loss_meter  = AverageMeter()
    accumulator = MetricAccumulator()
    pbar = tqdm(loader, desc=f'[ Val ] Epoch {epoch}', leave=False)
    for batch in pbar:
        pre    = batch['pre_img'].to(device, non_blocking=True)
        post   = batch['post_img'].to(device, non_blocking=True)
        loc_gt = batch['loc_mask'].to(device, non_blocking=True)
        dmg_gt = batch['dmg_mask'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            loc_logits, dmg_logits = model(pre, post)
            loss, _ = criterion(loc_logits, dmg_logits, loc_gt, dmg_gt)

        loss_meter.update(loss.item())
        accumulator.update(loc_logits, dmg_logits, loc_gt, dmg_gt)
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    return loss_meter.avg, accumulator.compute()


def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg.get('seed', 42))

    data_root     = Path(cfg['data']['root'])
    train_img_dir = data_root / 'train' / 'images'
    train_lbl_dir = data_root / 'train' / 'labels'
    exp_dir  = Path(cfg['experiment']['dir'])
    ckpt_dir = exp_dir / 'checkpoints'
    log_dir  = exp_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    img_size = cfg['data']['img_size']
    nw       = cfg['data']['num_workers']

    all_ds  = XBDDataset(str(train_img_dir), str(train_lbl_dir), transform=None)
    n_total = len(all_ds)
    n_val   = int(n_total * cfg['data']['val_split'])
    n_train = n_total - n_val

    g   = torch.Generator().manual_seed(42)
    idx = torch.randperm(n_total, generator=g).tolist()
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    train_ds = Subset(
        XBDDataset(str(train_img_dir), str(train_lbl_dir), transform=get_train_transforms(img_size)),
        train_idx
    )
    val_ds = Subset(
        XBDDataset(str(train_img_dir), str(train_lbl_dir), transform=get_val_transforms(img_size)),
        val_idx
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                              shuffle=True,  num_workers=nw, pin_memory=(nw > 0),
                              drop_last=True, persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'],
                              shuffle=False, num_workers=nw, pin_memory=(nw > 0),
                              persistent_workers=(nw > 0))

    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} samples')

    model = SiameseUNet(
        encoder_name=cfg['model']['encoder'],
        encoder_weights=cfg['model']['encoder_weights'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: {n_params/1e6:.1f}M')

    criterion = CombinedLoss(w_loc=cfg['loss']['w_loc'], w_dmg=cfg['loss']['w_dmg'])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['epochs'],
        eta_min=cfg['training']['lr'] * 0.01
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    writer = SummaryWriter(log_dir=str(log_dir))

    best_score = 0.0
    epochs = cfg['training']['epochs']

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        val_loss, metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        score = metrics['xview2_score']
        print(
            f'Epoch {epoch:3d}/{epochs} | '
            f'Train: {train_loss:.4f} | Val: {val_loss:.4f} | '
            f'F1_loc: {metrics["f1_loc"]:.4f} | '
            f'F1_dmg: {metrics["f1_dmg_macro"]:.4f} | '
            f'Score: {score:.4f}'
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Metrics/F1_loc',   metrics['f1_loc'],       epoch)
        writer.add_scalar('Metrics/F1_dmg',   metrics['f1_dmg_macro'], epoch)
        writer.add_scalar('Metrics/Score',    score,                   epoch)
        for name in ['no-damage', 'minor-damage', 'major-damage', 'destroyed']:
            writer.add_scalar(f'Metrics/F1_{name}', metrics.get(f'f1_{name}', 0), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        ckpt = {
            'epoch':       epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'score':       score,
            'metrics':     metrics,
            'config':      cfg,
        }

        if epoch % cfg['training'].get('save_every', 5) == 0:
            torch.save(ckpt, ckpt_dir / f'epoch_{epoch:03d}.pth')
        if score > best_score:
            best_score = score
            torch.save(ckpt, ckpt_dir / 'best_model.pth')
            print(f'  New best: {best_score:.4f}')

    writer.close()
    print(f'\nDone. Best Score: {best_score:.4f}')
    print(f'Checkpoint: {ckpt_dir / "best_model.pth"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/baseline.yaml')
    args = parser.parse_args()
    main(args.config)