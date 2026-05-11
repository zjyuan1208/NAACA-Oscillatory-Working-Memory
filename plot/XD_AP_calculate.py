import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def analyze_confusion_matrix(file_path):
    """
    note
    """
    violence_codes = ['B1', 'B2', 'B4', 'B5', 'B6', 'G']
    
    # confusion matrix.
    confusion = {gt: {pred: 0 for pred in violence_codes + ['None', 'Miss']} 
                 for gt in violence_codes}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('[source'):
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            filename, prediction_str = parts[0], parts[1]
            
            # Implementation note.
            gt_match = re.search(r'_label_([A-Z0-9-]+)', filename)
            gt_labels = set()
            if gt_match:
                gt_part = gt_match.group(1)
                gt_labels = {code for code in violence_codes if code in gt_part}
            
            # Implementation note.
            pred_labels = {p.strip() for p in prediction_str.split(',') 
                          if p.strip() != 'None'}
            
            # Implementation note.
            for gt in gt_labels:
                if gt in pred_labels:
                    confusion[gt][gt] += 1
                else:
                    if len(pred_labels) == 0:
                        confusion[gt]['Miss'] += 1
                    else:
                        for pred in pred_labels:
                            if pred in violence_codes:
                                confusion[gt][pred] += 1

    # confusion matrix.
    print("=" * 80)
    print(f"Confusion Matrix for: {os.path.basename(file_path)}")
    print("=" * 80)
    
    header = "GT / Pred"
    print(f"{header:<10}", end='')
    for pred in violence_codes + ['Miss']:
        print(f"{pred:>8}", end='')
    print(f"  {'Total':>8}")
    print("-" * 80)
    
    for gt in violence_codes:
        print(f"{gt:<10}", end='')
        total = sum(confusion[gt].values())
        for pred in violence_codes + ['Miss']:
            count = confusion[gt][pred]
            if count > 0:
                pct = count / total * 100 if total > 0 else 0
                print(f"{count:3d}({pct:4.1f}%)", end='')
            else:
                print(f"{'':>11}", end='')
        print(f"  {total:>8}")
    
    print("=" * 80)
    
    # Generate.
    generate_heatmap(confusion, violence_codes, file_path)
    
    return confusion


def create_custom_colormap():
    """
    Create custom colormap based on provided 6 colors.
    Color progression: deep blue -> blue -> light blue-gray -> light pink -> red -> deep red

    Returns:
        LinearSegmentedColormap: Custom colormap for heatmap visualization
    """
    # Define 6 colors in RGB values (normalized to 0-1 range)
    colors = [
        (11 / 255, 66 / 255, 94 / 255),    # deep blue
        (93 / 255, 139 / 255, 157 / 255),  # blue
        (202 / 255, 213 / 255, 219 / 255), # light blue-gray
        (224 / 255, 198 / 255, 197 / 255), # light pink
        (144 / 255, 69 / 255, 80 / 255),   # red
        (86 / 255, 19 / 255, 25 / 255)     # deep red
    ]

    # Create custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_heatmap', colors, N=256)

    return custom_cmap


def generate_heatmap(confusion, violence_codes, file_path):
    """
    note
    """
    # label mapping.
    label_names = {
        'B1': 'B1\nFighting',
        'B2': 'B2\nShooting',
        'B4': 'B4\nRiot',
        'B5': 'B5\nAbuse',
        'B6': 'B6\nCar Accident',
        'G': 'G\nExplosion'
    }
    # **Label Explanation**
    # B1: Fighting
    # B2: Shooting
    # B4: Riot
    # B5: Abuse
    # B6: Car accident
    # G: Explosion

    
    # percentage data.
    pred_categories = violence_codes + ['Miss']
    matrix = np.zeros((len(violence_codes), len(pred_categories)))
    
    for i, gt in enumerate(violence_codes):
        total = sum(confusion[gt].values())
        for j, pred in enumerate(pred_categories):
            count = confusion[gt][pred]
            matrix[i, j] = (count / total * 100) if total > 0 else 0
    
    # custom.
    custom_cmap = create_custom_colormap()
    
    # create figure.
    plt.figure(figsize=(15, 12))
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap=custom_cmap,
                xticklabels=[label_names.get(p, p) for p in pred_categories],
                yticklabels=[label_names[gt] for gt in violence_codes],
                cbar_kws={'label': 'Percentage (%)'},
                annot_kws={'fontsize': 22},
                linewidths=0.5, linecolor='gray',
                vmin=0, vmax=100)
    
    # plt.title('Confusion Matrix - Percentage (%)', fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=22, fontweight='bold')
    plt.ylabel('Ground Truth Label', fontsize=22, fontweight='bold')
    
    # Implementation note.
    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    
    # highlight diagonal.
    # for i in range(len(violence_codes)):
    #     ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
    #                                 edgecolor='blue', lw=3))
    
    plt.tight_layout()
    
    # Save.
    output_path = file_path.replace('.txt', '_confusion_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap saved to: {output_path}")
    
    plt.show()


# Run analysis.
confusion = analyze_confusion_matrix(
    '/home/zhyuan/Desktop/PCD/plot/BioWM_p2.txt'
)