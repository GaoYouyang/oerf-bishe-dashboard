"""Plot independently verified conditional-head aggregates, never weights."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_world_component_head_20260907'


def main():
    data = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    x = np.arange(5)
    labels = ['p14-s05','p22-s03','p33-s01','p45-s05','p58-s03']
    fig,axes = plt.subplots(1,2,figsize=(12,5.3))
    for shift,key,label,color in ((-.18,'previous','Frozen parent','#2876a3'),(.18,'query','Nine-coefficient head','#b63c4b')):
        axes[0].bar(x+shift,[100*r[key]['p90'][0] for r in data['rows']],width=.34,label=label,color=color)
    axes[0].axhline(1,color='#303030',linestyle='--',label='1% accuracy gate')
    axes[0].set_ylim(0,40)
    axes[0].set_ylabel('Held-out post-K1 field p90 error (%)')
    axes[0].set_title('Field accuracy is almost unchanged',fontsize=11)
    axes[0].legend(frameon=False,fontsize=9)
    bars = axes[1].bar(x,[100*r['relative_train_improvement'] for r in data['rows']],width=.6,color='#537b57')
    axes[1].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    axes[1].set_ylim(0,.20)
    axes[1].set_ylabel('Relative reduction in pre-K1 train objective (%)')
    axes[1].set_title('Conditional optimum adds less than 0.16%',fontsize=11)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=20)
        ax.set_xlabel('Held-out trajectory defining the fold')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A global world-component matrix does not solve the accuracy gap',fontsize=13)
    fig.text(.5,.025,'Opened clean proxy | 0/505 pass | 405 frames harmed vs parent | reporting repair disclosed',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.23,left=.07,right=.98,wspace=.34)
    fig.savefig(ROOT/f'assets/figures/{STEM}.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
