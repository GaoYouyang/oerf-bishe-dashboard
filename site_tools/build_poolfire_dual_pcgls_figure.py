"""Plot both saved numerical endpoints, not deployment timing."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_dual_pcgls_20260907'
d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
assert d['accurate_per_path'] == [0, 0]
values = np.array(d['scores'])*100
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for k, (ax, label, color) in enumerate(zip(axes.flat,
        ('Field', 'Full gradient', 'Interior gradient', 'Observation'),
        ('#b84654', '#207873', '#696174', '#3c718f'), strict=True)):
    for p, (marker, style, title) in enumerate((('o', '-', 'Field PCGLS'), ('x', '--', 'Right CGLS'))):
        ax.plot(np.arange(1, 6), values[p, :, k], marker=marker, linestyle=style,
                color=color, label=title, markersize=6)
    ax.axhline(1, color='#333333', linewidth=1, linestyle=':', label='1% gate')
    ax.set_title(label, fontsize=12)
    ax.set_ylabel('Relative error (%)')
    ax.set_xlabel('Opened sentinel')
    ax.set_xticks(np.arange(1, 6))
    ax.set_ylim(0, max(1.2, float(values[:, :, k].max())*1.2))
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc='best', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Frozen-map preconditioning: 0/5 within classical action caps\nTwo paths; sealed endpoints audited after parent serialization failure', fontsize=13)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png', dpi=160)
plt.close(fig)
