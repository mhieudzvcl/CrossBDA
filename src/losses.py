"""
losses.py - Loss functions for Building Damage Assessment

Strategy: Focal Loss + Dice Loss for both localization and damage heads.
Class weights handle severe imbalance in xBD (background ~55%, no-damage ~25%, rare classes <10%).
Total = w_loc * (FocalLoc + DiceLoc) + w_dmg * (FocalDmg + DiceDmg)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: -alpha * (1 - pt)^gamma * log(pt)
    Reduces the relative loss for easy examples, focusing on hard ones.
    """
    def __init__(self, gamma: float = 2.0, weight=None, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob     = torch.exp(log_prob)

        targets_clamped = targets.clone()
        mask = None
        if self.ignore_index >= 0:
            mask = targets == self.ignore_index
            targets_clamped[mask] = 0

        log_pt = log_prob.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)
        pt     = prob.gather(1, targets_clamped.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.weight is not None:
            w     = self.weight.to(logits.device)
            alpha = w[targets_clamped]
        else:
            alpha = 1.0

        loss = -alpha * focal_weight * log_pt

        if mask is not None and mask.any():
            loss = loss.masked_fill(mask, 0.0)
            return loss.sum() / (~mask).float().sum().clamp(min=1)

        return loss.mean()


class DiceLoss(nn.Module):
    """Soft Dice Loss averaged over all classes (macro). Effective for rare classes."""
    def __init__(self, smooth: float = 1.0, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        prob = F.softmax(logits, dim=1)

        valid_mask = (targets != self.ignore_index)
        t = targets.clone()
        t[~valid_mask] = 0

        one_hot = F.one_hot(t, num_classes=num_classes).permute(0, 3, 1, 2).float()

        vm      = valid_mask.unsqueeze(1).float()
        prob    = prob    * vm
        one_hot = one_hot * vm

        inter = (prob * one_hot).sum(dim=(0, 2, 3))
        union = (prob + one_hot).sum(dim=(0, 2, 3))
        dice  = (2 * inter + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice loss for both output heads.
    Weights estimated from xBD pixel frequency statistics.
    """
    DAMAGE_WEIGHTS = torch.tensor([0.2, 0.5, 2.0, 2.5, 3.0], dtype=torch.float32)
    LOC_WEIGHTS    = torch.tensor([0.3, 1.5], dtype=torch.float32)

    def __init__(
        self,
        focal_gamma: float = 2.0,
        w_loc: float = 0.5,
        w_dmg: float = 0.5,
        w_focal: float = 0.6,
        w_dice: float = 0.4,
    ):
        super().__init__()
        self.w_loc   = w_loc
        self.w_dmg   = w_dmg
        self.w_focal = w_focal
        self.w_dice  = w_dice

        self.loc_focal = FocalLoss(gamma=focal_gamma, weight=self.LOC_WEIGHTS)
        self.loc_dice  = DiceLoss()
        self.dmg_focal = FocalLoss(gamma=focal_gamma, weight=self.DAMAGE_WEIGHTS)
        self.dmg_dice  = DiceLoss()

    def forward(self, loc_logits, dmg_logits, loc_targets, dmg_targets):
        l_focal = self.loc_focal(loc_logits, loc_targets)
        l_dice  = self.loc_dice(loc_logits, loc_targets)
        loc_loss = self.w_focal * l_focal + self.w_dice * l_dice

        d_focal = self.dmg_focal(dmg_logits, dmg_targets)
        d_dice  = self.dmg_dice(dmg_logits, dmg_targets)
        dmg_loss = self.w_focal * d_focal + self.w_dice * d_dice

        total = self.w_loc * loc_loss + self.w_dmg * dmg_loss
        return total, {
            'loc_focal': l_focal.item(),
            'loc_dice':  l_dice.item(),
            'dmg_focal': d_focal.item(),
            'dmg_dice':  d_dice.item(),
            'loc_loss':  loc_loss.item(),
            'dmg_loss':  dmg_loss.item(),
        }