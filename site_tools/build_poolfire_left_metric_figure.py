"""Plot certificate action counts and the unresolved warm-start attribution."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_left_metric_20260907'
d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
assert d['status'] == 'INCONCLUSIVE_LEFT_METRIC_CONTROL_DOMINANCE'
x = np.arange(1, 6)


def counts(path, arm, action):
    return np.array([next(q['online'][action] for q in d['costs']
                         if q['path'] == path and q['arm'] == arm and q['fold'] == f)
                     for f in range(5)])


fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
axes[0].plot(x, d['classic_best_A'], 's-', color='#666666', label='Best classical iterative control')
for path, (marker, style, name) in enumerate((('o', '-', 'Formal'), ('x', '--', 'Independent'))):
    for arm, color, label in (('learned_metric', '#19796e', 'Warm + learned metric'),
                              ('zero_learned_metric', '#bb4d68', 'Zero + learned metric')):
        axes[0].plot(x, counts(path, arm, 'A'), marker=marker, linestyle=style,
                     color=color, label=f'{label}: {name}', markersize=6)
    for action, color, offset in (('A', '#19796e', -.07), ('AT', '#bb4d68', .07)):
        delta = counts(path, 'learned_metric', action)-counts(path, 'zero_learned_metric', action)
        axes[1].plot(x+offset, delta, marker=marker, linestyle=style, color=color,
                     label=f'{action}: {name}', markersize=6)
axes[0].set(title='Certified A calls: metric helps', ylabel='A calls', ylim=(0, 1050))
axes[1].set(title='Warm minus zero: attribution unresolved', ylabel='Action-count difference',
            ylim=(-2.7, 1.8), yticks=[-2, -1, 0, 1])
axes[1].axhline(0, color='#444444', linewidth=1, linestyle=':')
for ax in axes:
    ax.set_xlabel('Opened sentinel')
    ax.set_xticks(x)
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc='best', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Same sufficient four-metric certificate, five opened clean sentinels\n'
             'Not minimum true-accuracy calls or measured wall time; zero A(0) is avoidable', fontsize=12)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
