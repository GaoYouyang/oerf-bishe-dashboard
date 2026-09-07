"""Plot public aggregate errors separately from conservative certificates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_teacher_fidelity_warm_20260908'
data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
x = np.arange(5)
actual = [100*p['actual_error_max'][2] for p in data['points']]
bound = [100*p['certificate_max'][2] for p in data['points']]
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
first = ax.bar(x-.18, actual, width=.36, color='#167d94', label='Actual interior-gradient error')
second = ax.bar(x+.18, bound, width=.36, color='#a55855', label='Certificate upper bound')
ax.axhline(1, color='#242424', linestyle='--', linewidth=1.3, label='1% requirement')
ax.bar_label(first, labels=[f'{v:.3f}%' for v in actual], padding=3, fontsize=11)
ax.bar_label(second, labels=[f'{v:.3f}%' for v in bound], padding=3, fontsize=11)
ax.set_xticks(x, [f'Point {i+1}' for i in x])
ax.set_ylabel('Relative error or bound (%)')
ax.set_ylim(0, 1.95)
ax.set_axisbelow(True)
ax.grid(axis='y', alpha=.18)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper center', ncol=1, fontsize=10, frameon=False)
ax.set_title('Accurate-teacher warm start: actual accuracy passes, certified savings do not\nFive opened midpoints; larger value across two implementations', fontsize=14)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
