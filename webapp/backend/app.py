import os, io, base64, collections
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

# Model definition (must match Kaggle training exactly)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights=None, n_dmg=5):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        ec = self.encoder.out_channels
        self.bottleneck = DoubleConv(ec[-1] * 2, 512)
        skip_chs = list(reversed([c * 2 for c in ec[1:-1]]))
        dec_out = [256, 128, 64, 32]
        self.decoder = nn.ModuleList()
        inc = 512
        for s, o in zip(skip_chs, dec_out):
            self.decoder.append(DecoderBlock(inc, s, o))
            inc = o
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = DoubleConv(32, 32)
        self.loc_head = nn.Conv2d(32, 2, 1)
        self.dmg_head = nn.Conv2d(32, n_dmg, 1)

    def forward(self, pre, post):
        pf = self.encoder(pre)
        qf = self.encoder(post)
        x = self.bottleneck(torch.cat([pf[-1], qf[-1]], dim=1))
        skips = [torch.cat([pf[i], qf[i]], dim=1) for i in range(len(pf) - 2, 0, -1)]
        for blk, sk in zip(self.decoder, skips):
            x = blk(x, sk)
        x = self.final_conv(self.final_up(x))
        return self.loc_head(x), self.dmg_head(x)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = SiameseUNet(encoder_name='resnet34', encoder_weights=None).to(device)
ckpt_path = r"H:\KhoaLuan\experiments\baseline_resnet34\checkpoints\best_model.pth"
if os.path.exists(ckpt_path):
    print(f"Loading weights from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'], strict=False)
    print("Weights loaded OK")
else:
    print(f"WARNING: checkpoint not found at {ckpt_path}")
model.eval()

# Color map for damage classes
DAMAGE_COLORS = {
    0: [20, 20, 20],      # background - dark
    1: [34, 197, 94],     # no damage  - green
    2: [234, 179, 8],     # minor      - yellow
    3: [249, 115, 22],    # major      - orange
    4: [239, 68, 68],     # destroyed  - red
}
CLASS_NAMES = {1: 'No Damage', 2: 'Minor Damage', 3: 'Major Damage', 4: 'Destroyed'}

def decode_mask(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, v in DAMAGE_COLORS.items():
        rgb[mask == k] = v
    return Image.fromarray(rgb)

def compute_stats(loc_mask: np.ndarray, dmg_mask: np.ndarray) -> dict:
    """Compute pixel counts and percentages for each damage class."""
    building_pixels = int((loc_mask > 0).sum())
    total_pixels = loc_mask.size
    stats = {"building_coverage_pct": round(building_pixels / total_pixels * 100, 1)}
    counts = {}
    if building_pixels > 0:
        for cls_id, name in CLASS_NAMES.items():
            cnt = int((dmg_mask == cls_id).sum())
            counts[name] = {
                "pixels": cnt,
                "pct": round(cnt / building_pixels * 100, 1)
            }
    stats["damage_breakdown"] = counts
    return stats

def preprocess(img_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((512, 512), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

@app.get("/")
def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/api/predict")
async def predict(pre: UploadFile = File(...), post: UploadFile = File(...)):
    pre_tensor  = preprocess(await pre.read())
    post_tensor = preprocess(await post.read())

    with torch.no_grad():
        with torch.amp.autocast(device.type):
            loc_logits, dmg_logits = model(pre_tensor, post_tensor)

    loc_mask = loc_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    dmg_mask = dmg_logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Only show damage where building was detected
    dmg_mask[loc_mask == 0] = 0

    stats = compute_stats(loc_mask, dmg_mask)

    result_img = decode_mask(dmg_mask)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {"prediction_base64": b64, "stats": stats}