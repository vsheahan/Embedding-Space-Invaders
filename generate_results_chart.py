#!/usr/bin/env python3
"""
Generate a visual chart of the Embedding Space Invaders results.
Creates a PNG showing the spectacular failure metrics.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# SEP Dataset Results
sep_metrics = {
    'Detection\nRate': 96.6,
    'False Positive\nRate': 96.9,
    'Accuracy': 34.8
}

colors_sep = ['#2ecc71', '#e74c3c', '#e74c3c']  # Green for detection, red for bad metrics
bars1 = ax1.bar(sep_metrics.keys(), sep_metrics.values(), color=colors_sep, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, (metric, value) in zip(bars1, sep_metrics.items()):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax1.set_title('SEP Dataset Results\n"It flags EVERYTHING"', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylim(0, 105)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random Guess (50%)')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add annotation
ax1.annotate('😱', xy=(1, 96.9), xytext=(1.5, 85),
            fontsize=40, ha='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Jailbreak Dataset Results
jailbreak_metrics = {
    'Detection\nRate': 100.0,
    'False Positive\nRate': 100.0,
    'True\nNegatives': 0.0
}

colors_jb = ['#2ecc71', '#e74c3c', '#e74c3c']
bars2 = ax2.bar(jailbreak_metrics.keys(), jailbreak_metrics.values(), color=colors_jb, alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bar, (metric, value) in zip(bars2, jailbreak_metrics.items()):
    height = bar.get_height()
    if value == 0:
        label_text = '0\n(Not a\nsingle one!)'
        y_pos = 5
    else:
        label_text = f'{value:.0f}%'
        y_pos = height

    ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
            label_text,
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Jailbreak Dataset Results\n"Perfect disaster"', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylim(0, 105)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

# Add annotation
ax2.annotate('💀', xy=(1, 100), xytext=(1.5, 85),
            fontsize=40, ha='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Overall title
fig.suptitle('👾 Embedding Space Invaders: A Visual Summary of Spectacular Failure',
             fontsize=16, fontweight='bold', y=0.98)

# Add footer
fig.text(0.5, 0.02, 'Translation: The system flags almost everything as malicious, regardless of whether it actually is.',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])

# Save
plt.savefig('assets/results_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Chart saved to assets/results_chart.png")

# Also create a confusion matrix visualization
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

# SEP Confusion Matrix
sep_cm = np.array([[7, 223],    # TN=7, FP=223 (adjusted for visualization)
                   [4, 114]])    # FN=4, TP=114
sep_cm_normalized = sep_cm.astype('float') / sep_cm.sum(axis=1)[:, np.newaxis] * 100

im1 = ax3.imshow(sep_cm, cmap='RdYlGn_r', alpha=0.8, vmin=0, vmax=250)
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Predicted\nSafe', 'Predicted\nAttack'], fontsize=11)
ax3.set_yticklabels(['Actually\nSafe', 'Actually\nAttack'], fontsize=11)
ax3.set_title('SEP Dataset Confusion Matrix\n(Ground Truth)', fontsize=13, fontweight='bold', pad=15)

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, f'{sep_cm[i, j]}\n({sep_cm_normalized[i, j]:.1f}%)',
                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')

# Jailbreak Confusion Matrix
jb_cm = np.array([[0, 141],     # TN=0, FP=141 (ALL safe flagged!)
                  [0, 261]])     # FN=0, TP=261
jb_cm_normalized = jb_cm.astype('float')
jb_cm_normalized[jb_cm_normalized == 0] = np.nan

im2 = ax4.imshow(jb_cm, cmap='RdYlGn_r', alpha=0.8, vmin=0, vmax=300)
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Predicted\nSafe', 'Predicted\nAttack'], fontsize=11)
ax4.set_yticklabels(['Actually\nSafe', 'Actually\nAttack'], fontsize=11)
ax4.set_title('Jailbreak Dataset Confusion Matrix\n(Everything flagged!)', fontsize=13, fontweight='bold', pad=15)

# Add text annotations
for i in range(2):
    for j in range(2):
        value = jb_cm[i, j]
        if value == 0:
            text_str = '0\n😱'
        else:
            text_str = f'{value}\n(100%)'
        text = ax4.text(j, i, text_str,
                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')

fig2.suptitle('👾 Confusion Matrices: Where It All Went Wrong',
              fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('assets/confusion_matrices.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Confusion matrices saved to assets/confusion_matrices.png")

print("\n🎨 Visualizations generated! Add to README with:")
print("![Results](assets/results_chart.png)")
print("![Confusion Matrices](assets/confusion_matrices.png)")
