"""Plot public aggregates without raw fields, weights or parameter probes."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_tensor_train_diagnosis_20260907'


def main():
    data = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    labels = ['p14-s05', 'p22-s03', 'p33-s01', 'p45-s05', 'p58-s03']
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    for offset, group, label, color in ((-.18, 'train', 'Train frames', '#007f70'),
                                       (.18, 'query', 'Held-out frames', '#c83242')):
        axes[0].bar(x+offset, [100*r[group+'_post']['p90'][0] for r in data['rows']],
                    width=.34, label=label, color=color)
    axes[0].axhline(1, linestyle='--', color='#333333', label='1% gate')
    axes[0].set_ylabel('Post-K1 field p90 relative error (%)')
    axes[0].set_ylim(0, 40)
    axes[0].legend(frameon=False, fontsize=9)
    values = [100*r['relative_train_objective_descent'] for r in data['rows']]
    axes[1].bar(x, values, color='#2866ad', width=.58)
    axes[1].set_ylabel('Pre-K1 train-objective reduction (%)')
    axes[1].set_ylim(0, .04)
    for xx, value in zip(x, values):
        axes[1].text(xx, value+.001, f'{value:.4f}%', ha='center', fontsize=9)
    for ax, title in zip(axes, ('Training error is already substantial', 'Tiny train-only descent probes')):
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, labels, rotation=20)
        ax.set_xlabel('Held-out trajectory defining the fold')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frozen model diagnosis: local descent does not establish 1% accuracy', fontsize=13)
    fig.text(.5, .025, 'Opened clean proxy | 0 optimizer updates | probes never deployed | no new accuracy pass',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.23, left=.07, right=.98, wspace=.3)
    fig.savefig(ROOT/f'assets/figures/{STEM}.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
