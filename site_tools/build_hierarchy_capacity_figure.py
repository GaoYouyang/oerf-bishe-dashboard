import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'docs/poolfire_hierarchy_capacity_2026-10-07_public_summary.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=100)
fig.patch.set_facecolor('white')
colors = ['#16836b', '#c84145', '#62686a']
axes[0].bar(['Full factor\nPCGLS34', 'Fixed hierarchy\nPCGLS34', 'Plain\nCGLS35'],
            [data['exact_factor_strict_matches'], data['primary_strict_matches'], 0], color=colors)
for j, value in enumerate([99, 0, 0]):
    axes[0].text(j, value + 2, f'{value}/99', ha='center')
axes[0].set_ylim(0, 114)
axes[0].set_ylabel('Four-metric matched cells')
axes[0].set_title('Accuracy loss on opened sentinels', fontsize=13)
storage = [data['dense_inverse_factor_bytes_by_camera_count']['5'] / 2**20,
           data['compressed_factor_bytes_by_camera_count']['5'] / 2**20]
axes[1].bar(['Dense inverse\nfactor array', 'Compressed\nfactor array'], storage, color=colors[:2])
for j, value in enumerate(storage):
    axes[1].text(j, value + 22, f'{value:.1f} MiB', ha='center')
axes[1].set_ylim(0, 1240)
axes[1].set_ylabel('Factor array storage (MiB)')
axes[1].set_title('Smaller array, failed accuracy', fontsize=13)
for axis in axes:
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=.15)
    axis.set_axisbelow(True)
fig.text(.5, .025, 'No learned result or resource speedup; setup and solver memory excluded. Native 5/7/9 cameras, three frames per trajectory.',
         ha='center', fontsize=9)
fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.18, wspace=.28)
fig.savefig(ROOT / 'assets/figures/poolfire_hierarchy_capacity_20261007.png')
plt.close(fig)
