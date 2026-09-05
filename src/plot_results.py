import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

methods = ["Baseline", "FDA\nInference", "FDA\nAug", "FS 5%\n(LP)", "FS 10%\n(LP)", "Hybrid\nFDA+FT 10%"]
f1_loc  = [0.6904,    0.6647,        0.7454,    0.6904,      0.6904,      0.7527]
f1_dmg  = [0.1683,    0.1598,        0.2485,    0.1695,      0.1707,      0.2042]
xview2  = [0.3250,    0.3112,        0.3982,    0.3258,      0.3266,      0.3687]

x = np.arange(len(methods))
w = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - w, f1_loc, w, label="F1 Localization", color="#4c72b0")
b2 = ax.bar(x,     f1_dmg, w, label="F1 Damage",       color="#dd8452")
b3 = ax.bar(x + w, xview2, w, label="xView2 Score",    color="#55a868")

ax.set_ylabel("Score")
ax.set_title("Domain Adaptation Ablation Study on ida-BD (Test = 77 images)")
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9)
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.6)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, rotation=45)

fig.tight_layout()
os.makedirs(r"H:\KhoaLuan\experiments\plots", exist_ok=True)
out = r"H:\KhoaLuan\experiments\plots\ablation_study.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")