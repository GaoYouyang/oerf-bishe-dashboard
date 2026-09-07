"""Plot sanitized actual-cost intervals and the distinct angular descriptor."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_warm_cost_angular_audit_20260908'
data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout='constrained', width_ratios=(1.4, 1))
colors = ['#177a69', '#386d9c', '#b16a3b', '#765c83']
arms = ['nonlinear_metric', 'zero_metric', 'scalar_metric', 'k4_teacher']
labels = ['Nonlinear warm', 'Zero + same metric', 'Scalar warm', 'K4 dual-ridge']
for a, (arm, label, color) in enumerate(zip(arms, labels, colors)):
    ranges = np.array([r['intervals']['first_cost'][arm] for r in data['points']])
    centers = ranges.mean(axis=1)
    axes[0].errorbar(np.arange(1, 6)+(a-1.5)*.15, centers, yerr=(ranges[:, 1]-ranges[:, 0])/2,
        fmt='o', markersize=5, capsize=4, color=color, label=label)
axes[0].set_xticks(range(1, 6))
axes[0].set_xlabel('Five reused trajectory midpoints')
axes[0].set_ylabel('A calls, and separately the same number of adjoints')
axes[0].set_ylim(116, 198)
axes[0].set_title('Actual-cost gate: 0 robust wins, 4 failures, 1 unclear', fontsize=11)
axes[0].legend(loc='upper left', frameon=False, ncol=2, fontsize=8.5)
for a, (row, color) in enumerate(zip(data['angular']['descriptors']['angular_descriptors'], ['#386d9c', '#b16a3b', '#177a69'])):
    value = row['median']
    axes[1].errorbar(a, value, yerr=[[value-row['minimum']], [row['maximum']-value]],
        fmt='o', color=color, markersize=7, capsize=6)
    axes[1].annotate(f'{value:,.0f}', (a, value), xytext=(8, 6), textcoords='offset points', fontsize=9)
axes[1].axhline(1, color='#555555', linestyle='--', linewidth=1, label='Zero-estimate control E = 1')
axes[1].set_yscale('log')
axes[1].set_ylim(.4, 1e6)
axes[1].set_xlim(-.6, 2.9)
axes[1].set_xticks(range(3), ['Unit', 'Jacobi', 'Learned'])
axes[1].set_ylabel('Squared relative observation-error ratio E (log scale)')
axes[1].set_title('Omitted-view error: attribution, not a cost win', fontsize=11)
axes[1].legend(loc='center left', frameon=False, fontsize=9)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle('Two independent retrospective audits; no new warm-start benefit', fontsize=14)
fig.supxlabel('Left: truth-visible crossings, not a deployable stop. Right: median and range over old endpoints, not field-error multipliers.', fontsize=9)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
