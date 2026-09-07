"""Render the published aggregate errors, never private experiment inputs."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_world_tensor1840_20260907'


def main():
    data = json.loads((ROOT / f'docs/{STEM}.json').read_text())
    reports = {r['arm']: r for r in data['reports']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    labels = ['p14-s05', 'p22-s03', 'p33-s01', 'p45-s05', 'p58-s03']
    arms = [('world_tensor1840', 'Learned tensor 1840', '#c83242'),
            ('fixed_world_tensor', 'Fixed tensor', '#767676'),
            ('direct_field_ridge', 'Cheaper field ridge', '#b67808'),
            ('rayset369', 'Existing learned 369', '#007f70'),
            ('full_direct', 'Full direct', '#2866ad')]
    for ax, metric, title in zip(axes, (0, 2), ('Field', 'Interior gradient')):
        for arm, label, color in arms:
            values = [100*r['p90'][metric] for r in reports[arm]['trajectories']]
            ax.plot(labels, values, color=color, marker='o', linewidth=1.7, label=label)
        ax.axhline(1, color='#222222', linestyle='--', linewidth=1, label='1% gate')
        ax.set_title(title, fontsize=12)
        ax.set_ylim(-2, 100)
        ax.grid(axis='y', alpha=0.2)
        ax.tick_params(axis='x', rotation=20)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Trajectory p90 relative error (%)')
    fig.suptitle('Trained 1840-parameter model: 0/505 at the four-metric 1% gate', fontsize=14)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5,0.06))
    fig.text(0.5,0.02,'Opened PoolFire trajectories | clean nine-camera proxy | not external validation',
             ha='center', fontsize=10)
    fig.subplots_adjust(top=.86, bottom=.27, left=.07, right=.98, wspace=.12)
    fig.savefig(ROOT/f'assets/figures/{STEM}.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
