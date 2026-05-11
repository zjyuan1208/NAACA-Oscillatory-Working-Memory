import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Implementation note.
plt.style.use('seaborn-v0_8-whitegrid')

# custom.
colors = [
    (11 / 255, 66 / 255, 94 / 255),    # deep blue
    (93 / 255, 139 / 255, 157 / 255),  # blue
    (202 / 255, 213 / 255, 219 / 255), # light blue-gray
    (224 / 255, 198 / 255, 197 / 255), # light pink
    (144 / 255, 69 / 255, 80 / 255),   # red
    (86 / 255, 19 / 255, 25 / 255)     # deep red
]

colors = {
    'dataset1': colors[-2], # Implementation note.
    'dataset2': colors[1], # Implementation note.
    'warning': colors[-3], # Implementation note.
    'dark': colors[0], # Implementation note.
}

# Implementation note.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Implementation note.
json_path1 = "/home/zhyuan/Desktop/PCD/plot/dataset_xd_cls.json"
print(f"Implementation note.")
with open(json_path1, 'r') as f:
    data1 = json.load(f)

# Implementation note.
ratios1 = []
for sample in data1:
    total_length = sample['total_length']
    time_sent = sample['time_sent_to_qwen']
    if total_length > 0:
        ratio = time_sent / total_length
        ratios1.append(ratio)

ratios1 = np.array(ratios1)

print(f"Implementation note.")
print(f"Implementation note.")
print(f"Implementation note.")

# Implementation note.
json_path2 = "/home/zhyuan/Desktop/Qwen-Audio/results/biooss_multimetric_pattern_change_results.json"
print(f"Implementation note.")
with open(json_path2, 'r') as f:
    data2 = json.load(f)

# Implementation note.
ratios2 = []
for sample in data2['results']:
    time_sent = sample['time_length_sent_to_qwen']
    total_length = 60.0 # Audio.
    ratio = time_sent / total_length
    ratios2.append(ratio)

ratios2 = np.array(ratios2)

print(f"Implementation note.")
print(f"Implementation note.")
print(f"Implementation note.")

# Implementation note.
# Implementation note.
fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
ax.set_facecolor('white')

# Implementation note.
datasets = [
    {'ratios': ratios1, 'center': 0.7, 'color': colors['dataset1'], 'cmap': 'Reds', 'name': 'XD-Violence'},
    {'ratios': ratios2, 'center': 1.3, 'color': colors['dataset2'], 'cmap': 'Blues', 'name': 'USoW'}
]

for dataset in datasets:
    ratios = dataset['ratios']
    center = dataset['center']
    color = dataset['color']
    cmap = dataset['cmap']
    
    # Implementation note.
    kde = stats.gaussian_kde(ratios)
    y_range = np.linspace(0, 1.1, 200)
    density = kde(y_range)
    density_normalized = density / density.max() * 0.18 # Implementation note.
    
    ax.fill_betweenx(y_range, center - density_normalized, center, 
                      alpha=0.6, color=color, 
                      edgecolor=color, linewidth=1.5)
    
    # Implementation note.
    bp = ax.boxplot([ratios], positions=[center], widths=0.08, # Implementation note.
                     vert=True, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor=color, alpha=0.8, linewidth=1.5),
                     medianprops=dict(color=colors['warning'], linewidth=2),
                     whiskerprops=dict(color=colors['dark'], linewidth=1.3, linestyle='-'),
                     capprops=dict(color=colors['dark'], linewidth=1.3))
    
    # Implementation note.
    np.random.seed(42 + int(center * 10))
    x_jitter = np.random.normal(center + 0.13, 0.03, size=len(ratios)) # Implementation note.
    
    point_colors = []
    for ratio in ratios:
        idx = np.argmin(np.abs(y_range - ratio))
        density_val = density[idx]
        color_intensity = density_val / density.max()
        point_colors.append(color_intensity)
    
    scatter = ax.scatter(x_jitter, ratios, c=point_colors, cmap=cmap, 
                        s=12, alpha=0.5, edgecolors='white', linewidth=0.15, zorder=3) # Implementation note.

# Statistics.
# Implementation note.
q1_1, q2_1, q3_1 = np.percentile(ratios1, [25, 50, 75])
mean_1 = ratios1.mean()

stats_text_1 = f"""XD-Violence (N={len(ratios1)})
Mean: {mean_1:.3f}
Median: {q2_1:.3f}
Q1-Q3: {q1_1:.3f}-{q3_1:.3f}"""

ax.text(0.7, 0.05, stats_text_1, fontsize=7, # Implementation note.
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                 edgecolor=colors['dataset1'], linewidth=1.2, alpha=0.95),
        verticalalignment='bottom', horizontalalignment='center',
        fontfamily='monospace')

# Implementation note.
q1_2, q2_2, q3_2 = np.percentile(ratios2, [25, 50, 75])
mean_2 = ratios2.mean()

stats_text_2 = f"""USoW (N={len(ratios2)})
Mean: {mean_2:.3f}
Median: {q2_2:.3f}
Q1-Q3: {q1_2:.3f}-{q3_2:.3f}"""

ax.text(1.3, 0.05, stats_text_2, fontsize=7, # Implementation note.
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                 edgecolor=colors['dataset2'], linewidth=1.2, alpha=0.95),
        verticalalignment='bottom', horizontalalignment='center',
        fontfamily='monospace')

# Implementation note.
ax.set_ylim(0, 1.1)
ax.set_xlim(0.3, 1.7) # Implementation note.
ax.set_ylabel('Time Sent Ratio', fontweight='bold', fontsize=10)
# ax.set_title('Comparison of Time Sent Ratios\nBetween Two Datasets', 
# Implementation note.
ax.set_xticks([0.7, 1.3])
ax.set_xticklabels(['XD-Violence', 'USoW'], fontsize=9, fontweight='bold')

# Grid.
ax.grid(False)
# ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Implementation note.
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['dataset1'], alpha=0.7, label='XD-Violence', 
          edgecolor=colors['dataset1'], linewidth=1.5),
    Patch(facecolor=colors['dataset2'], alpha=0.7, label='USoW', 
          edgecolor=colors['dataset2'], linewidth=1.5)
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, 
          edgecolor='white', fontsize=8, title='Datasets', title_fontsize=9)

plt.tight_layout()

# Save.
output_path = '/home/zhyuan/Desktop/PCD/plot/comparison_time_sent.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Save.")

# Implementation note.
print("\n" + "="*70)
print("Implementation note.")
print("="*70)
print(f"Implementation note.")
print("-"*70)
print(f"Implementation note.")
print(f"Implementation note.")
print(f"Implementation note.")
print(f"Implementation note.")
print(f"Implementation note.")
print(f"Implementation note.")
print(f"{'Q1 (25%)':<20} {np.percentile(ratios1, 25):<25.4f} {np.percentile(ratios2, 25):<25.4f}")
print(f"{'Q3 (75%)':<20} {np.percentile(ratios1, 75):<25.4f} {np.percentile(ratios2, 75):<25.4f}")

# Implementation note.
from scipy.stats import mannwhitneyu
statistic, pvalue = mannwhitneyu(ratios1, ratios2, alternative='two-sided')
print(f"Parameters.")
print(f"Implementation note.")
print(f"  p-value: {pvalue:.6f}")
if pvalue < 0.001:
    print(f"Implementation note.")
elif pvalue < 0.01:
    print(f"Implementation note.")
elif pvalue < 0.05:
    print(f"Implementation note.")
else:
    print(f"Implementation note.")

print("="*70)

plt.show()