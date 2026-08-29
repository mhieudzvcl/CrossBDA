"""
metrics.py - Danh gia F1, IoU cho Building Damage Assessment
Theo chuan xView2 Challenge:
  - F1-localization: F1 tren binary building mask
  - F1-damage:       Macro F1 tren 4 damage classes (chi tinh tren building pixels)
  - Overall score:   0.3 * F1_loc + 0.7 * F1_dmg
"""
import numpy as np
import torch

DAMAGE_CLASS_NAMES = ['background', 'no-damage', 'minor-damage', 'major-damage', 'destroyed']

class AverageMeter:
    """Tinh trung binh cong don."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / max(self.count, 1)

class MetricAccumulator:
    """
    Tich luy True Positives, False Positives, False Negatives qua cac batch
    de tinh metric cuoi epoch ma khong bi tran RAM.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        # Localization (binary)
        self.loc_tp = 0
        self.loc_fp = 0
        self.loc_fn = 0
        
        # Damage (4 classes: 1, 2, 3, 4)
        self.dmg_tp = {1: 0, 2: 0, 3: 0, 4: 0}
        self.dmg_fp = {1: 0, 2: 0, 3: 0, 4: 0}
        self.dmg_fn = {1: 0, 2: 0, 3: 0, 4: 0}

    def update(
        self,
        loc_logits:  torch.Tensor,
        dmg_logits:  torch.Tensor,
        loc_targets: torch.Tensor,
        dmg_targets: torch.Tensor,
    ):
        """Nhan logits (B, C, H, W) va targets (B, H, W)."""
        loc_pred = loc_logits.argmax(dim=1).cpu().numpy()   # (B, H, W)
        dmg_pred = dmg_logits.argmax(dim=1).cpu().numpy()
        loc_gt   = loc_targets.cpu().numpy()
        dmg_gt   = dmg_targets.cpu().numpy()

        # --- Tich luy Localization ---
        self.loc_tp += np.sum((loc_pred == 1) & (loc_gt == 1))
        self.loc_fp += np.sum((loc_pred == 1) & (loc_gt == 0))
        self.loc_fn += np.sum((loc_pred == 0) & (loc_gt == 1))

        # --- Tich luy Damage ---
        # xView2 chi tinh diem damage tren nhung pixel thuoc ve toa nha (loc_gt == 1)
        building_mask = (loc_gt == 1)
        if np.sum(building_mask) > 0:
            dmg_p = dmg_pred[building_mask]
            dmg_t = dmg_gt[building_mask]
            
            for k in [1, 2, 3, 4]:
                self.dmg_tp[k] += np.sum((dmg_p == k) & (dmg_t == k))
                self.dmg_fp[k] += np.sum((dmg_p == k) & (dmg_t != k))
                self.dmg_fn[k] += np.sum((dmg_p != k) & (dmg_t == k))

    def compute(self) -> dict:
        results = {}
        eps = 1e-8
        
        # --- F1 Localization ---
        f1_loc = 2 * self.loc_tp / (2 * self.loc_tp + self.loc_fp + self.loc_fn + eps)
        results['f1_loc'] = f1_loc

        # --- F1 Damage ---
        f1_dmg_classes = []
        for k in [1, 2, 3, 4]:
            tp, fp, fn = self.dmg_tp[k], self.dmg_fp[k], self.dmg_fn[k]
            f1_k = 2 * tp / (2 * tp + fp + fn + eps) if (tp + fp + fn) > 0 else 0.0
            results[f'f1_{DAMAGE_CLASS_NAMES[k]}'] = f1_k
            f1_dmg_classes.append(f1_k)
            
        f1_dmg_macro = np.mean(f1_dmg_classes)
        results['f1_dmg_macro'] = f1_dmg_macro

        # --- Overall xView2 Score ---
        results['xview2_score'] = 0.3 * f1_loc + 0.7 * f1_dmg_macro

        return results