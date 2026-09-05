"""Plot the redacted, independently validated source budget, without stacking."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / 'docs/real_bost_error_sources_v280_public_summary.json').read_text())
    assert data['independent_validation']['passed']
    colors = ['#52636b', '#007a78', '#c13c54', '#7955a4']
    labels = ['Omitted structure', 'Inverse aliasing', 'Geometry mismatch', 'Observation noise']
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, space, title in zip(axes.flat, ['field', 'gradient', 'interior_gradient', 'observation'],
                               ['Field', 'Full gradient', 'Interior gradient', 'True-geometry observation'], strict=True):
        rows = [r for r in data['summaries'] if r['space'] == space]
        medians = np.array([r['signed_shares']['p50'] for r in rows])
        for j, label in enumerate(labels):
            # Break curves at condition boundaries; do not imply paired perturbations.
            for start in (0, 4, 8):
                ax.plot(np.arange(start, start+4), medians[start:start+4, j], 'o-',
                        color=colors[j], linewidth=1.7, markersize=4, label=label if start == 0 else None)
        ax.axhline(0, color='#99a3a6', linewidth=.8)
        for x in (3.5, 7.5):
            ax.axvline(x, color='#cbd3d4', linestyle=':', linewidth=1)
        ax.set_title(title, loc='left', fontsize=13)
        ax.set_ylim(-.08, 1.08)
        ax.set_xticks(range(12), ['0', '.25', '.75', '1']*3)
        ax.set_ylabel('Median signed contribution')
        ax.grid(axis='y', alpha=.18)
        ax.spines[['right', 'top']].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel('clean                         pose                         combined\nNormalized time within each condition')
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend, loc='upper center', bbox_to_anchor=(.5, .91), ncol=4, frameon=False)
    fig.suptitle('v280 | Where does the fixed coarse-model error come from?', x=.06, ha='left', fontsize=18)
    fig.text(.06, .035, '1,404 opened virtual cells; truth-visible attribution, not an algorithm gain.\n'
             'Quantiles are NOT additive percentages. Pose/combined geometry draws are not paired.', fontsize=10, color='#435258')
    fig.subplots_adjust(left=.075, right=.98, bottom=.16, top=.83, hspace=.25, wspace=.2)
    fig.savefig(ROOT / 'assets/figures/real_bost_error_sources_v280.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
