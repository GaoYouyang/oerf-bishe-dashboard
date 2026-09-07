"""Render an offline counterfactual diagnosis, never a deployment speed plot."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'docs/poolfire_warm_cost_angular_audit_20260908.json'
OUT = ROOT/'assets/figures/poolfire_warm_error_shape_20260908.png'


def main():
    data = json.loads(DATA.read_text())['error_shape']
    assert data['status'] == 'CONSISTENT_ERROR_DIRECTION_PENALTY'
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    fig.suptitle('Same error size, different refinement difficulty', fontsize=16, y=.98)
    fig.text(.5, .91, 'Five opened midpoints; two independent arithmetic paths', ha='center', fontsize=10)
    pairs = [('scaled_cold', 'warm', 'Error norm fixed at the original warm level'),
             ('cold', 'restored_warm', 'Error norm fixed at the original zero-start level')]
    for ax, (cold, warm, title) in zip(axes, pairs):
        for name, offset, color, label in ((cold, -.08, '#178579', 'Zero-start error direction'),
                                         (warm, .08, '#ba4b53', 'Warm error direction')):
            vals = np.array([p['intervals']['first'][name] for p in data['points']])
            mid = vals.mean(axis=1)
            ax.errorbar(np.arange(1, 6)+offset, mid, yerr=(vals[:, 1]-vals[:, 0])/2,
                fmt='o', linestyle='none', color=color, markersize=6, capsize=4, label=label)
            for j, (low, high) in enumerate(vals):
                text = str(int(low)) if low == high else f'{int(low)}-{int(high)}'
                ax.annotate(text, (j+1+offset, mid[j]), xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, color=color)
        ax.set_title(title, fontsize=10, pad=12)
        ax.set_xticks(range(1, 6), [f'P{j}' for j in range(1, 6)])
        ax.set_xlabel('Opened midpoint')
        ax.set_ylim(0, 240)
        ax.set_xlim(.5, 5.5)
        ax.grid(axis='y', color='#dddddd', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Refinement steps to all four 1% errors')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .87), ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .075, 'Left: zero-direction state is artificial. Right: warm-direction state is artificial.', ha='center', fontsize=9)
    fig.text(.5, .04, 'Both require a full reference. Not a deployable initialization, online cost or measured speedup.', ha='center', fontsize=9)
    fig.subplots_adjust(top=.75, bottom=.19, left=.08, right=.98, wspace=.13)
    fig.savefig(OUT, dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
