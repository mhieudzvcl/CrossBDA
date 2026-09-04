import matplotlib.pyplot as plt
import numpy as np
import os

# Data
methods = ['Baseline', 'FDA Aug', '5% FT', '10% FT']
f1_loc = [0.6904, 0.7454, 0.6904, 0.6904]
f1_dmg = [0.1683, 0.2485, 0.1695, 0.1707]
xview2 = [0.3250, 0.3975, 0.3258, 0.3266]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, f1_loc, width, label='F1 Localization', color='#4c72b0')
rects2 = ax.bar(x, f1_dmg, width, label='F1 Damage', color='#dd8452')
rects3 = ax.bar(x + width, xview2, width, label='xView2 Score', color='#55a868')

ax.set_ylabel('Scores')
ax.set_title('Domain Adaptation Performance on ida-BD (Test set = 77)')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, rotation=45)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
os.makedirs(r'H:\KhoaLuan\experiments\plots', exist_ok=True)
out_p = r'H:\KhoaLuan\experiments\plots\exp3_comparison.png'
plt.savefig(out_p, dpi=300, bbox_inches='tight')
print(f'Saved to {out_p}')