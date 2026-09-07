"""Render only public, independently verified aggregate stage diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT/'docs/poolfire_camera_mix_ablation_20260907.json').read_text())['stage_attribution']
rows = data['trajectories']
fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.35, 1]})
x = np.arange(5)
ax.bar(x-.18, [100*r['pre_gap_p50'][2] for r in rows], .36, color='#4877a2', label='Before K1')
ax.bar(x+.18, [100*r['post_gap_p50'][2] for r in rows], .36, color='#b65748', label='After K1')
ax.axhline(0, color='#303030', lw=.8)
ax.set_xticks(x, ['p14', 'p22', 'p33', 'p45', 'p58'])
ax.set_ylabel('Median interior-gradient gap (percentage points)')
ax.set_title('Full mixing minus self-only\nPositive: full mixing is worse', fontsize=12)
ax.legend(frameon=False, fontsize=10)
ax.grid(axis='y', alpha=.2)
labels = ['Harm before and after', 'New after K1', 'Removed by K1', 'Neither stage']
counts = data['stage_counts'][2]
bars = bx.barh(np.arange(4), counts, color=['#b65748', '#d99a55', '#45958b', '#7f8c98'])
bx.set_yticks(np.arange(4), labels)
bx.invert_yaxis()
bx.set_xlim(0, 280)
bx.bar_label(bars, padding=4)
bx.set_xlabel('Frames (505 total)')
bx.set_title('Interior-gradient harm stage\nFixed models, not algorithm improvement', fontsize=12)
for a in (ax, bx):
    a.spines[['top', 'right']].set_visible(False)
fig.suptitle('Camera mixing: the p33 trade-off already exists in the initializer', fontsize=14)
fig.tight_layout(rect=(0, .04, 1, .91))
fig.text(.5, .015, 'Opened clean nine-camera proxy | No refit | Both versions fail four-metric 1% accuracy', ha='center', fontsize=10)
fig.savefig(ROOT/'assets/figures/poolfire_camera_mix_stage_20260907.png', dpi=160)
plt.close(fig)
