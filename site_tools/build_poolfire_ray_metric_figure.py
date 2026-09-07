"""Plot only independently validated public aggregate comparisons."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_ray_metric_tensor_20260907'


def main():
    data = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    x = np.arange(5)
    labels = ['p14-s05','p22-s03','p33-s01','p45-s05','p58-s03']
    fig,axes = plt.subplots(1,2,figsize=(12,5.3))
    for shift,key,label,color in ((-.18,'previous','Prior L-BFGS','#2876a3'),(.18,'query','Local metric + L-BFGS','#b63c4b')):
        axes[0].bar(x+shift,[100*r[key]['p90'][0] for r in data['rows']],width=.34,label=label,color=color)
    axes[0].axhline(1,color='#303030',linestyle='--',label='1% accuracy gate')
    axes[0].set_ylim(0,40)
    axes[0].set_ylabel('Held-out post-K1 field p90 error (%)')
    axes[0].set_title('Four of five field tails worsen',fontsize=11)
    axes[0].legend(frameon=False,fontsize=9)
    bars = axes[1].bar(x,[r['any_harm_vs_previous'] for r in data['rows']],width=.6,color='#6c7665')
    axes[1].bar_label(bars,padding=3)
    axes[1].set_ylim(0,115)
    axes[1].set_ylabel('Frames with at least one metric harmed / 101')
    axes[1].set_title('459 harmed frames versus the prior learned model',fontsize=11)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=20)
        ax.set_xlabel('Held-out trajectory defining the fold')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same 1840 parameters and training budget: local normalization fails',fontsize=13)
    fig.text(.5,.025,'Opened clean proxy | 0/505 accuracy pass | three new cheap controls retained | no speed claim',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.23,left=.07,right=.98,wspace=.32)
    fig.savefig(ROOT/f'assets/figures/{STEM}.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
