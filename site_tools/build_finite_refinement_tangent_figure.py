import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SITE = Path(__file__).resolve().parents[1]
DATA = 'poolfire_finite_refinement_tangent_2026-10-09_public_summary.json'
OUTPUT = 'poolfire_finite_refinement_tangent_2026-10-09.png'


def main():
    d = json.loads((SITE / 'docs' / DATA).read_text())
    assert d['independent_checks_passed'] == d['independent_checks_total'] == 22
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.80, wspace=.32)
    labels = ['All', '5 cameras', '7 cameras', '9 cameras']
    good = [d['final_locally_improving']] + [d['by_camera'][str(c)]['final_improving'] for c in (5, 7, 9)]
    bad = [d['final_locally_worsening']] + [d['by_camera'][str(c)]['final_worsening'] for c in (5, 7, 9)]
    x = np.arange(4)
    left.bar(x, good, color='#237b64', label='Locally improves')
    left.bar(x, bad, bottom=good, color='#bf4852', label='Locally worsens')
    for j, (g, b) in enumerate(zip(good, bad)):
        left.text(j, g / 2, str(g), ha='center', va='center', color='white', weight='bold')
        left.text(j, g + b + 2.5, f'{b} worsen', ha='center', va='bottom', fontsize=9)
    left.set_ylim(0, 115)
    left.set_xticks(x, labels)
    left.set_ylabel('Number of fixed anchors')
    left.set_title('Derivative after 28 unchanged steps', fontsize=12)
    left.legend(loc='upper right', frameon=False, fontsize=9)
    rem = d['tangent_remainder']
    values = [rem[k] for k in ('p50', 'p90_higher', 'max')]
    right.bar(['p50', 'p90', 'Worst'], values, color=['#3b6b9a', '#6a5794', '#b58130'], width=.6)
    for j, v in enumerate(values):
        right.text(j, v + .012, f'{v:.4f}', ha='center', fontsize=10)
    right.set_ylim(0, .40)
    right.set_ylabel('Endpoint remainder / tangent norm')
    right.set_title('Finite endpoint is not its first tangent', fontsize=12)
    for ax in (left, right):
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e1e4e7', linewidth=.7)
    fig.suptitle('Local benefit does not establish finite-step matched accuracy', fontsize=15, y=.95)
    fig.text(.5, .105, '99 opened-train anchors; raw field distance to a finite teacher, not CFD truth error.', ha='center', fontsize=10)
    fig.text(.5, .065, 'No amplitude search, refit, effective call savings or new predictor authorization.', ha='center', fontsize=10)
    fig.savefig(SITE / 'assets' / OUTPUT, dpi=170, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
