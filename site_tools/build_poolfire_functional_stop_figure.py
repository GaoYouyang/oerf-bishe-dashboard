"""Plot the two independently measured primary-minus-Jacobi call counts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_functional_stop_20260907'
data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
assert data['status'] == 'FAIL_FUNCTIONAL_CERTIFIED_WARM_NECESSARY_PILOT'
primary, jacobi = data['reports'][0], data['reports'][3]
deltas = np.array([[a['A']-b['A'] for a,b in zip(primary['costs'][p],jacobi['costs'][p])] for p in range(2)])
fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
x = np.arange(5)
ax.bar(x-.17, deltas[0], width=.32, color='#207873', label='Formal')
ax.bar(x+.17, deltas[1], width=.32, color='#b84654', label='Independent')
ax.axhline(0, color='#303030', linewidth=1)
ax.set_xticks(x, ['P14', 'P22', 'P33', 'P45', 'P58'])
ax.set_ylabel('Extra forward A calls relative to Zero-Jacobi')
ax.set_xlabel('One opened midpoint per trajectory; not full sequences')
ax.set_title('Certified accuracy holds; no stable warm-start savings', fontsize=15, pad=14)
ax.legend(frameon=False)
ax.grid(axis='y', alpha=.16)
ax.set_axisbelow(True)
for p, offset in enumerate((-.17, .17)):
    for i, value in enumerate(deltas[p]):
        ax.text(i+offset, value+(.35 if value >= 0 else -.35), f'{value:+d}', ha='center',
                va='bottom' if value >= 0 else 'top', fontsize=10)
ax.set_ylim(min(deltas.min()-3, -5), deltas.max()+5)
for name in ('top', 'right'):
    ax.spines[name].set_visible(False)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
