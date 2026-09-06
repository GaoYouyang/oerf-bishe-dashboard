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
fig, ax = plt.subplots(figsize=(14.4, 6.2), layout='constrained')
x = np.arange(5)
colors = ('#283e51', '#d69730', '#a14771', '#197e73', '#426fa8', '#ad5446')
for offset, (key, label), color in zip((-.35, -.21, -.07, .07, .21, .35),
        (('graph_message_k1', 'Cross-camera graph, 177 parameters'),
         ('local_message_k1', 'Same-camera graph, 177 parameters'),
         ('trained_ray_set_k1', 'Earlier fully trained, 369 parameters'),
         ('ray_set_k1', 'Fixed random features'), ('linear_response_k1', 'Linearized control'),
         ('field_k1', 'Cheaper direct-field ridge')), colors):
    values = [r['passing_cells'] for r in arms[key]['trajectory_results']]
    bars = ax.bar(x + offset, values, width=.125, color=color, label=label)
    ax.bar_label(bars, padding=3, fontsize=8)
ax.axhline(101, color='#596269', linestyle='--', linewidth=1)
ax.set_ylim(0, 145)
ax.set_xticks(x, [row['trajectory'] for row in arms['ray_set_k1']['trajectory_results']])
ax.set_ylabel('Cells passing all four relative-error gates / 101')
ax.set_title('Same 505 PoolFire frames, complete-trajectory holdout', loc='left', pad=44, fontsize=17)
ax.text(0, 1.045, 'Paired graph result: cross-camera 367/505, 1/5 trajectories | same-camera 370/505, 1/5', transform=ax.transAxes, fontsize=12)
ax.legend(loc='upper center', ncol=2, frameon=False, fontsize=9)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=.16)
ax.set_axisbelow(True)
fig.supxlabel('Post-open clean geometry. Graphs add work beyond 2A+2AT. Dashed: all 101 required. No speedup or real BOST.', fontsize=10)
fig.savefig(ROOT / f'assets/figures/{STEM}.png', dpi=170, metadata={'Title': 'PoolFire fixed F30 aggregate comparison'})
plt.close(fig)
