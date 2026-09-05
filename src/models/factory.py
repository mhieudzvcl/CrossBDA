import sys
from pathlib import Path

# Add src to path so we can import model.py and models.scale_mae_unet
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import SiameseUNet
from models.scale_mae_unet import SiameseScaleMAE

def create_model(config):
    """
    Factory function to create the model based on the config.
    """
    model_name = config.get('model', {}).get('encoder', 'resnet34')
    
    if model_name == 'scalemae':
        print("Using Scale-MAE backbone")
        return SiameseScaleMAE(
            num_damage_classes=5,
            vit_model="vit_base_patch16",
            input_res=1.0 # Will be updated later via dataset/config if needed
        )
    else:
        print(f"Using {model_name} backbone (SiameseUNet)")
        return SiameseUNet(
            encoder_name=model_name,
            encoder_weights=config.get('model', {}).get('encoder_weights', 'imagenet'),
            num_damage_classes=5
        )
