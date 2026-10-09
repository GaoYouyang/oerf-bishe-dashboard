import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SITE = Path(__file__).resolve().parents[1]
data = json.loads((SITE / 'docs/poolfire_global_review_2026-10-09_public_summary.json').read_text())
values = data['p90_error_ratios_to_reference']
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
colors = ('#ba4c57', '#227c87', '#51615a')
names = ('Camera grouped', 'Global directions', 'Equal-budget CGLS')
positions = np.arange(3)
for ax, metric, title in zip(axes, ('field', 'observation'), ('Field error', 'Observation error')):
    for j, (key, name, color) in enumerate(zip(('grouped', 'global', 'cold'), names, colors)):
        ax.bar(positions + (j - 1) * .24, values[f'{key}_{metric}'], width=.22, label=name, color=color)
    ax.axhline(1., color='#303030', linestyle='--', linewidth=1., label='CGLS128 reference')
    ax.set_xticks(positions, ['5', '7', '9'])
    ax.set_xlabel('Active cameras')
    ax.set_ylabel('p90 CFD error / reference error')
    ax.set_title(title, fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e2e6e4')
axes[0].legend(fontsize=8, loc='upper left')
axes[0].set_ylim(0, 1.82)
axes[1].set_ylim(0, 5.45)
fig.suptitle('99 opened train anchors: grouped directions do not beat the controls', fontsize=12)
fig.savefig(SITE / 'assets/poolfire_global_review_2026-10-09.png', dpi=180)
plt.close(fig)
