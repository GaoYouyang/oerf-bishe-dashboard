"""Plot only the redacted aggregate; no research data or models are read."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_f30_comparison_20260906'
data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
arms = {row['arm']: row for row in data['arms']}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
fig, ax = plt.subplots(figsize=(12.8, 5.6), layout='constrained')
x = np.arange(5)
colors = ('#a14771', '#197e73', '#426fa8', '#ad5446')
for offset, (key, label), color in zip((-.30, -.10, .10, .30),
        (('trained_ray_set_k1', 'Fully trained, 369 parameters'),
         ('ray_set_k1', 'Fixed random features'), ('linear_response_k1', 'Linearized control'),
         ('field_k1', 'Cheaper direct-field ridge')), colors):
    values = [r['passing_cells'] for r in arms[key]['trajectory_results']]
    bars = ax.bar(x + offset, values, width=.18, color=color, label=label)
    ax.bar_label(bars, padding=3, fontsize=10)
ax.axhline(101, color='#596269', linestyle='--', linewidth=1)
ax.set_ylim(0, 132)
ax.set_xticks(x, [row['trajectory'] for row in arms['ray_set_k1']['trajectory_results']])
ax.set_ylabel('Cells passing all four relative-error gates / 101')
ax.set_title('Same 505 PoolFire frames, complete-trajectory holdout', loc='left', pad=44, fontsize=17)
ax.text(0, 1.045, 'Complete trajectories: fully trained 1/5 | random features 2/5 | linear 1/5 | ridge 0/5', transform=ax.transAxes, fontsize=12)
ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=.16)
ax.set_axisbelow(True)
fig.supxlabel('Post-open clean fixed geometry. Dashed line: all 101 required. No stable speedup or real-BOST claim.', fontsize=10)
fig.savefig(ROOT / f'assets/figures/{STEM}.png', dpi=170, metadata={'Title': 'PoolFire fixed F30 aggregate comparison'})
plt.close(fig)
