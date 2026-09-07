"""Render only redacted, independently validated trajectory summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_full_metric_cost_20260908'
d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
assert d['status'] == 'PASS_FULL_TRAJECTORY_METRIC_COST'
rows = d['trajectories']
x = np.arange(1, 6)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
axes[0].plot(x, [r['candidate_A']['p50'] for r in rows], 'o-', color='#19796e', label='Certified learned metric: median')
axes[0].plot(x, [r['oracle_A_lower']['p50'] for r in rows], 's-', color='#666666', label='Ideal Jacobi: median')
axes[0].set(title='All 505 frames accurate; both action counts lower', ylabel='A calls', ylim=(0, 520))
axes[1].bar(x-.18, [100*r['A_reduction']['p50'] for r in rows], width=.36, color='#19796e', label='Median saving')
weak = [100*r['worst_A_reduction'] for r in rows]
axes[1].bar(x+.18, weak, width=.36, color='#bd4d69', label='Weakest frame saving')
for i, value in enumerate(weak):
    axes[1].annotate(f'{value:.2f}%', (x[i]+.18, value), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
axes[1].set(title='Weak margins remain visible', ylabel='A reduction (%)', ylim=(0, 33))
for ax in axes:
    ax.set(xlabel='Opened trajectory (101 frames each)', xticks=x)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
    ax.legend(loc='upper right', fontsize=8, frameon=False)
fig.suptitle('Five complete trajectories: a learned-metric result, not warm-start attribution\n'
             'Clean fixed-nine-camera data; not measured time or prospective generalization', fontsize=12)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
