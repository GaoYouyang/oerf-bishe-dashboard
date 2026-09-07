"""Render only the public aggregate, with accuracy distinct from certification."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_camera_subset_metric_20260908'
data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, layout='constrained')
labels, colors = ['Unit CGLS', 'Jacobi PCGLS', 'Learned metric'], ['#637782', '#bb6849', '#167b6f']
x = np.arange(2)
for ax, key, title in zip(axes, ('physical_passes', 'certified_passes'),
        ('Actual four-metric accuracy', 'Observation-only stopping certificate')):
    for a, (label, color) in enumerate(zip(labels, colors)):
        counts = [row['arms'][a][key] for row in data['counts']]
        bars = ax.bar(x+(a-1)*.24, counts, width=.22, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v}/5' for v in counts], padding=4, fontsize=10)
    ax.set_xticks(x, ['7 cameras', '5 cameras'])
    ax.set_ylim(0, 6)
    ax.set_yticks(range(6))
    ax.set_title(title, fontsize=12, pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Opened midpoint frames passing')
axes[0].legend(loc='upper left', frameon=False, fontsize=9)
fig.suptitle('No-refit camera-removal pilot: formal gate fails (2/10)', fontsize=15)
fig.supxlabel('Fixed 2048-step budget | Five reused frames per subset | Not full-trajectory or warm-start success', fontsize=10)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
