"""Plot public TRAIN-only neighbor counts; no private inputs or raw arrays."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_metric_similarity_20260908'


def main():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    fig, axes = plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    colors = ['#168096','#9b9d9e','#a35a58']
    left = np.zeros(2)
    for key,color in zip(('better','same','worse'),colors):
        values = np.array([data['counts'][k][key] for k in ('field','weak')])
        axes[0].barh([0,1],values,left=left,color=color,label=key.capitalize())
        for j,value in enumerate(values):
            if value:
                axes[0].text(left[j]+value/2,j,str(value),ha='center',va='center',color='white')
        left += values
    axes[0].set(yticks=[0,1],yticklabels=['Field error','Weak-direction error'],
                xlabel='Overlapping TRAIN pairs',xlim=(0,20),xticks=[0,5,10,15,20])
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,frameon=False)
    x = np.arange(1,6)
    axes[1].plot(x,data['fold_ratio_median'],'o-',color=colors[0],label='Median ratio')
    axes[1].plot(x,data['fold_ratio_worst'],'s-',color=colors[2],label='Worst ratio')
    axes[1].axhline(1,color='#333333',ls='--',lw=1,label='No change')
    axes[1].set(xlabel='Outer fold: TRAIN-only diagnostic',xticks=x,
                ylabel='Weak-neighbor error / raw-neighbor error',ylim=(.94,1.25))
    axes[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Closer fields do not reliably repair weak directions\n20 overlapping TRAIN pairs; not outer testing or solver replay',fontsize=13)
    fig.savefig(ROOT/'assets/figures'/f'{STEM}.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
