"""Plot independently validated public optimizer-comparison aggregates."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_tensor_lbfgs_20260907'


def main():
    data = json.loads((ROOT/f'docs/{STEM}.json').read_text())
    x = np.arange(5)
    labels = ['p14-s05','p22-s03','p33-s01','p45-s05','p58-s03']
    fig,axes = plt.subplots(1,2,figsize=(12,5.3))
    for shift,key,label,color in ((-.18,'adam_query','Adam','#2876a3'),(.18,'query','Full-batch L-BFGS','#b63c4b')):
        axes[0].bar(x+shift,[100*r[key]['p90'][0] for r in data['rows']],width=.34,label=label,color=color)
    axes[0].axhline(1,color='#303030',linestyle='--',label='1% accuracy gate')
    axes[0].set_ylim(0,40)
    axes[0].set_ylabel('Held-out post-K1 field p90 error (%)')
    axes[0].set_title('Five better tails, but accuracy still fails',fontsize=11)
    for shift,key,label,color in ((-.18,'adam_train_objective','Adam','#2876a3'),(.18,'lbfgs_train_objective','Full-batch L-BFGS','#b63c4b')):
        axes[1].bar(x+shift,[r[key] for r in data['rows']],width=.34,label=label,color=color)
    axes[1].set_ylim(0,.13)
    axes[1].set_ylabel('Pre-K1 train objective (dimensionless)')
    axes[1].set_title('Lower training loss is not a 1% pass',fontsize=11)
    for ax in axes:
        ax.set_xticks(x,labels,rotation=20)
        ax.set_xlabel('Held-out trajectory defining the fold')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Same 1840-parameter model: optimizer improvement is insufficient',fontsize=13)
    fig.text(.5,.025,'Opened clean proxy | 0/505 accuracy pass | 29 frames harmed vs Adam | all folds budget-limited',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.23,left=.07,right=.98,wspace=.3)
    fig.savefig(ROOT/f'assets/figures/{STEM}.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
