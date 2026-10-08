import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SITE = Path(__file__).resolve().parents[1]
NAME = 'poolfire_finite_loss_decomposition_2026-10-09'


def main():
    data = json.loads((SITE / 'docs' / (NAME + '_public_summary.json')).read_text())
    assert data['independent_checks_passed'] == data['independent_checks_total'] == 13
    rows = [dict(local_harm=data['local_harm'], quadratic_flip=data['local_benefit_quadratic_flip'],
                 remainder_flip=data['linearized_benefit_remainder_flip'], actual_benefit=data['actual_benefit'])]
    rows.extend(data['by_camera'][str(c)] for c in (5, 7, 9))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.subplots_adjust(left=.10, right=.97, bottom=.30, top=.82)
    x, bottom = np.arange(4), np.zeros(4)
    for key, label, color in [('local_harm', 'Locally harmful', '#8b5666'),
            ('quadratic_flip', 'Local benefit; linearized harm', '#ba7c30'),
            ('remainder_flip', 'Linearized benefit; actual harm', '#396f9f'),
            ('actual_benefit', 'Actual benefit', '#247a61')]:
        values = np.asarray([r[key] for r in rows])
        ax.bar(x, values, bottom=bottom, width=.6, label=label, color=color)
        for n, v in enumerate(values):
            if v >= 3:
                ax.text(n, bottom[n] + v / 2, str(v), ha='center', va='center', color='white', weight='bold')
        bottom += values
    for n, total in enumerate(bottom):
        ax.text(n, total + 2, f'{int(total)} anchors', ha='center', fontsize=10)
    ax.annotate('1 actual benefit', xy=(0, 98.5), xytext=(.45, 105),
                arrowprops=dict(arrowstyle='->', color='#247a61'), fontsize=9)
    ax.set_xticks(x, ['All', '5 cameras', '7 cameras', '9 cameras'])
    ax.set_ylim(0, 113)
    ax.set_ylabel('Number of fixed anchors')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e1e4e7', linewidth=.7)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False, fontsize=10)
    fig.suptitle('Local benefit can be lost before or after the solver remainder', fontsize=14, y=.94)
    fig.text(.5, .075, 'Mutually exclusive descriptive signs; raw field distance to a finite teacher only.', ha='center', fontsize=10)
    fig.text(.5, .035, '99 opened-train anchors; no amplitude search, new solve or causal failure percentages.', ha='center', fontsize=10)
    fig.savefig(SITE / 'assets' / (NAME + '.png'), dpi=170, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
