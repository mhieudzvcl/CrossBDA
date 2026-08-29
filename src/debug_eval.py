import torch
import numpy as np
import sys
sys.path.insert(0, r"H:\KhoaLuan")
from src.eval_ida import IdaDataset, SiameseUNet
from torch.utils.data import DataLoader

data_dir = r"H:\KhoaLuan\data\ida-BD\PRJ-3563\Project--ida-bd-pre-and-post-disaster-high-resolution-satellite-imagery-for-building-damage-assessment-from-hurricane-ida\data"
dataset = IdaDataset(data_dir)
pre, post, loc_true, dmg_true = dataset[0]

device = torch.device('cuda')
model = SiameseUNet().to(device)
model.load_state_dict(torch.load(r"H:\KhoaLuan\experiments\baseline_resnet34\checkpoints\best_model.pth", map_location=device)['model_state'], strict=False)
model.eval()

with torch.no_grad(), torch.amp.autocast('cuda'):
    loc_pred, dmg_pred = model(pre.unsqueeze(0).to(device), post.unsqueeze(0).to(device))
    
loc_pred_cls = loc_pred.argmax(1).squeeze(0).cpu().numpy()
loc_true_cls = loc_true.numpy()

print("True building pixels:", loc_true_cls.sum())
print("Pred building pixels:", loc_pred_cls.sum())