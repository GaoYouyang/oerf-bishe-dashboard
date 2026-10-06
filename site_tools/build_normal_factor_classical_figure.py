import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'docs/poolfire_normal_factor_classical_2026-10-07_public_summary.json').read_text())
assert data['independent_checks_passed'] == 24 and data['strict_matches']['NormalFactor-PCGLS34'] == 99
names = ['CGLS35', 'PCGLS35', 'KernelBank90-Warm33', 'NormalFactor-PCGLS34']
labels = ['CGLS35', 'Jacobi\nPCGLS35', 'Learned bank\nWarm33', 'Full-normal\nPCGLS34']
colors = ['#697580', '#578798', '#bc7159', '#348b68']
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=100)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e5e8ea')
heights = [data['strict_matches'][name] for name in names]
axes[0].bar(labels, heights, color=colors, width=.62)
axes[0].set_ylim(0, 116)
axes[0].set_ylabel('Four-metric matches / 99', fontsize=12)
axes[0].set_title('Opened three-frame train sentinels', fontsize=14, pad=15)
for j, height in enumerate(heights):
    axes[0].text(j, height + 3, str(height) + '/99', ha='center', fontsize=12)
budgets = [35, 35, 35, 34, 128]
axes[1].bar(labels + ['Finite\nCGLS128'], budgets, color=colors + ['#a2a8ad'], width=.62)
axes[1].set_ylim(0, 148)
axes[1].set_ylabel('A callbacks (AT nearly equal)', fontsize=12)
axes[1].set_title('Callback budgets, not elapsed time', fontsize=14, pad=15)
for j, value in enumerate(budgets):
    axes[1].text(j, value + 3, str(value), ha='center', fontsize=12)
fig.suptitle('Strong classical witness, not learned success or measured speedup', fontsize=16, y=.97)
fig.text(.5, .035, 'Full-normal arm also requires 34 factor solves; LU factors alone: 415-731 MiB per geometry. Setup is nonfree.',
         ha='center', fontsize=11)
fig.subplots_adjust(left=.065, right=.98, top=.81, bottom=.22, wspace=.28)
fig.savefig(ROOT / 'assets/figures/poolfire_normal_factor_classical_20261007.png', dpi=100)
plt.close(fig)
