import os, sys, io
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, r"H:\KhoaLuan")
try:
    from src.model import SiameseUNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    model = SiameseUNet(encoder_name='resnet34', encoder_weights=None).to(device)
    
    ckpt_path = r"H:\KhoaLuan\experiments\baseline_resnet34\checkpoints\best_model.pth"
    if os.path.exists(ckpt_path):
        print("Found best_model.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
    else:
        print("NOT FOUND best_model.pth")
        
    model.eval()

    # Create dummy images
    dummy_img = Image.new('RGB', (1024, 1024), color = 'black')
    buf = io.BytesIO()
    dummy_img.save(buf, format='PNG')
    img_bytes = buf.getvalue()

    def preprocess(img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((512, 512))
        arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        arr = (arr - mean) / std
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    pre_tensor = preprocess(img_bytes)
    post_tensor = preprocess(img_bytes)
    
    print("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast(device.type):
            _, dl = model(pre_tensor, post_tensor)
        preds = dl.argmax(1).squeeze(0).cpu().numpy()
    print("Inference success! Output shape:", preds.shape)
except Exception as e:
    import traceback
    traceback.print_exc()