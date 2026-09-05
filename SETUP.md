# Setup Guide for Collaborators

This guide helps you set up the full project environment after cloning.
After completing all steps, your directory structure will match the original.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/mhieudzvcl/CrossBDA.git
cd KhoaLuan
```

---

## Step 2: Create Conda Environment

```bash
conda create -n xbd_env python=3.10 -y
conda activate xbd_env
pip install -r requirements.txt
```

---

## Step 3: Download Data from Google Drive

Click the link below and download the entire `data` folder, then place it in the project root:

**Data folder (xBD + ida-BD + xBD-S12):**
https://drive.google.com/drive/folders/1R2etIZtg-454q1-9h9R8cXct4y0maC9E?usp=drive_link

After downloading, your directory should look like:
```
KhoaLuan/
  data/
    xBD/
      train/
      test/
      hold/
    ida-BD/
      images/
      masks/
      split/
        train_5/
        train_10/
        test/
    xBD-S12/
      s2_tci/
      s1/
      s2/
```

---

## Step 4: Download Pre-trained Models from Google Drive

Click the link below and download the entire `models` folder, then create the directory structure as below:

**Models folder (all checkpoints):**
https://drive.google.com/drive/folders/1fdio4tY0rmX8a0yilh9N7sx-bcHt9R1D?usp=drive_link

After downloading, place the files as follows:
```
KhoaLuan/
  experiments/
    baseline_resnet34/
      checkpoints/
        best_model.pth
    fda_results/
      fda_best_model.pth
    fewshot_results/
      fs5_model.pth
      fs10_model.pth
    hybrid_results/
      hybrid_fs10_model.pth
    LapSRN_x8.pb
```

Create the directories first (Windows):
```powershell
mkdir experiments\baseline_resnet34\checkpoints
mkdir experiments\fda_results
mkdir experiments\fewshot_results
mkdir experiments\hybrid_results
```

Or on Linux/Mac:
```bash
mkdir -p experiments/baseline_resnet34/checkpoints
mkdir -p experiments/fda_results experiments/fewshot_results experiments/hybrid_results
```

---

## Step 5: Verify Setup

Run a quick evaluation to confirm everything works:

```bash
conda activate xbd_env
python src/eval_ida.py --checkpoint experiments/baseline_resnet34/checkpoints/best_model.pth
```

Expected output:
```
xView2 Score     : 0.3250
F1 Localization  : 0.6904
F1 Damage (macro): 0.1683
```

If the output matches, your setup is complete.

---

## Step 6: (Optional) Run on Kaggle

Open `kaggle_fda_train.ipynb` and follow the instructions inside to run FDA Augmentation training on Kaggle GPU (Tesla T4).
