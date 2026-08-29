# Siamese U-Net - Building Damage Assessment (xBD)

## Setup
```bash
conda activate xbd_env
```

## Train
```bash
cd H:\KhoaLuan
python src/train.py --config configs/baseline.yaml
```

## Monitor (mo terminal moi)
```bash
tensorboard --logdir experiments/baseline_resnet34/logs
# http://localhost:6006
```

## Evaluate
```bash
python src/evaluate.py --config configs/baseline.yaml \
                        --checkpoint experiments/baseline_resnet34/checkpoints/best_model.pth \
                        --split test
```

## Cau truc du lieu xBD
```
data/xBD/
  train/images/  <event>_<id>_pre_disaster.png
                 <event>_<id>_post_disaster.png
  train/labels/  <event>_<id>_pre_disaster.json  (building polygons)
                 <event>_<id>_post_disaster.json  (building + damage label)
  test/...
  hold/...
```

## Label classes
| Value | Class        |
|-------|-------------|
| 0     | Background   |
| 1     | No-damage    |
| 2     | Minor-damage |
| 3     | Major-damage |
| 4     | Destroyed    |

## Metrics (xView2 chuẩn)
- F1-loc  : F1 binary building localization
- F1-dmg  : Macro F1 damage (4 classes, only on building pixels)
- Score   : 0.3 * F1_loc + 0.7 * F1_dmg
