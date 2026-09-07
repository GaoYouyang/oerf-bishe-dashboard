"""Plot independently verified public aggregate tails, never private parameters."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_butterfly_accuracy_20260907'
data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
arms = {r['arm']: r for r in data['reports']}
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)
for j, (ax, label) in enumerate(zip(axes.flat, ['Field', 'Full gradient', 'Interior gradient', 'Observation'])):
    for key, name, color, marker in [('butterfly_geometry243', 'Hierarchical model (243)', '#b65748', 'o'),
                                    ('geometry_diagonal86', 'Diagonal control (86)', '#358d80', 's'),
                                    ('world_tensor_lbfgs', 'Earlier L-BFGS (1840)', '#4877a2', '^')]:
        ax.plot(np.arange(5), [100*r['p90'][j] for r in arms[key]['trajectories']],
                marker=marker, color=color, lw=1.7, label=name)
    ax.axhline(1, color='#333333', linestyle='--', lw=1, label='Frozen 1% gate')
    ax.set_xticks(np.arange(5), ['p14', 'p22', 'p33', 'p45', 'p58'])
    ax.set_ylim(0, 50)
    ax.set_title(label, fontsize=12)
    ax.set_ylabel('Trajectory p90 relative error (%)')
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .04))
fig.suptitle('Hierarchical geometry model: fixed-budget accuracy gate failed', fontsize=14)
fig.tight_layout(rect=(0, .14, 1, .95))
fig.text(.5, .015, 'Opened clean 9-camera LOTO | Both new arms: 0/505 | Full direct: 505/505 | No convergence or speedup claim', ha='center', fontsize=9)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
