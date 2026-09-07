"""Plot the redacted squared-energy fractions; not additive components."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_inverse_moment_20260907'
data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
assert data['status'] == 'CONSISTENT_INVERSE_NORMAL_ERROR_RETENTION'
values = np.array([arm['ratios_median'] for arm in data['arms']])*100
fig, ax = plt.subplots(figsize=(10, 5.7), constrained_layout=True)
y = np.arange(3)
for k, (label, color) in enumerate(zip(('Inverse-normal weighted', 'Ordinary field', 'Forward image'),
                                      ('#b84654', '#207873', '#7b858c'), strict=True)):
    offset = (k-1)*.24
    bars = ax.barh(y+offset, values[:, k], height=.21, color=color, label=label)
    ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values[:, k]], padding=4, fontsize=10)
ax.set_yticks(y, ['L-BFGS warm start', 'Adam warm start', 'Normalized BP'])
ax.invert_yaxis()
ax.set_xlim(0, 111)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel('Median retained error energy vs zero initialization (%)\nDifferent weightings, not additive; 20 overlapping TRAIN pairs / 5 opened frames')
ax.set_title('Field error falls; weak-sensitivity-weighted error persists', fontsize=15, pad=15)
ax.legend(loc='lower right', frameon=False, fontsize=10)
ax.grid(axis='x', alpha=.16)
ax.set_axisbelow(True)
ax.margins(y=.20)
for name in ('top', 'right'):
    ax.spines[name].set_visible(False)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
