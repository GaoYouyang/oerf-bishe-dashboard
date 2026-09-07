"""Render validated aggregate cost comparisons; censoring is never a point cost."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_full_control_cost_20260908'
d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
assert d['status'] == 'PASS_FULL_ROSTER_CONTROL_COST'
by_arm = {r['arm']: r for r in d['cost_summaries']}
order = ('zero_cgls', 'normalized_bp', 'dual_ridge', 'fixed_metric')
labels = ['CGLS', 'BP + CGLS', 'Dual ridge', 'Untrained metric']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
for ax, key, title in zip(axes, ('median_A_saving_lower_bound', 'minimum_A_saving_lower_bound'),
    ('Median paired A reduction', 'Weakest-frame A reduction')):
    values = [100*by_arm[arm][key] for arm in order]
    bars = ax.bar(np.arange(4), values, color=['#19796e', '#497b91', '#a0616b', '#858585'], width=.6)
    bars[-1].set_hatch('//')
    for i, value in enumerate(values):
        prefix = '>=' if i == 3 else ''
        ax.annotate(f'{prefix}{value:.2f}%', (i, value), xytext=(0, 5),
            textcoords='offset points', ha='center', fontsize=10)
    ax.set(title=title, ylabel='A reduction (%)', xticks=np.arange(4), xticklabels=labels, ylim=(0, 51))
    ax.tick_params(axis='x', labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
fig.suptitle('505/505 frames beat all four controls in both A and AT\n'
             'Hatched bars: lower bounds from censored costs, not measured convergence', fontsize=12)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
