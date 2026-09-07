"""Independently verified counterfactual camera effects, not accuracy success."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_camera_mix_ablation_20260907'


def main():
    d = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    fig,axes = plt.subplots(1,2,figsize=(12,5.3))
    x = np.arange(4)
    bars = axes[0].bar(x-.18,d['full_metric_wins'],.34,label='Mixing better',color='#2876a3')
    axes[0].bar_label(bars,padding=3,fontsize=9)
    bars = axes[0].bar(x+.18,d['self_metric_wins'],.34,label='Self-only better',color='#b63c4b')
    axes[0].bar_label(bars,padding=3,fontsize=9)
    axes[0].set_xticks(x,['Field','Full gradient','Interior\ngradient','Observation'])
    axes[0].set_ylim(0,570)
    axes[0].set_ylabel('Better frames / 505')
    axes[0].legend(frameon=False,fontsize=9)
    axes[0].set_title('Mixing helps most field and observation errors',fontsize=11)
    x = np.arange(5)
    delta = [100*r['effect']['delta_p50'][2] for r in d['rows']]
    bars = axes[1].bar(x,delta,.6,color=['#2876a3' if v>0 else '#b63c4b' for v in delta])
    axes[1].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[1].axhline(0,color='#555555',linewidth=1)
    axes[1].set_xticks(x,['p14-s05','p22-s03','p33-s01','p45-s05','p58-s03'],rotation=20)
    axes[1].set_ylim(-.8,.65)
    axes[1].set_ylabel('Median interior-error difference (percentage points)')
    axes[1].set_title('Self-only minus mixing: positive favors mixing',fontsize=11)
    axes[1].set_xlabel('Held-out trajectory')
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Cross-camera exchange contributes, with a gradient trade-off',fontsize=13)
    fig.text(.5,.025,'Fixed trained features | each arm gain-matched on train only | both 0/505 at four-metric 1% accuracy',ha='center',fontsize=9)
    fig.subplots_adjust(top=.84,bottom=.23,left=.06,right=.98,wspace=.35)
    fig.savefig(ROOT/f'assets/figures/{STEM}.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
