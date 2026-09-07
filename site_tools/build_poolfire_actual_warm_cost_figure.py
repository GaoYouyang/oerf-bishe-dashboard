"""Plot only sealed, redacted cost intervals; no private inputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_actual_warm_cost_20260908'


def main():
    data = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
    fig, ax = plt.subplots(figsize=(10,5),layout='constrained')
    for offset,arm,label,color in (
        (-.18,'full_teacher','Accurate-teacher warm','#ad464c'),
        (0,'zero_metric','Zero + same learned metric','#167d82'),
        (.18,'k4_teacher','K4-teacher warm','#75608a')):
        bounds = np.array([p['intervals']['first_cost'][arm] for p in data['points']])
        y = bounds.mean(axis=1)
        ax.errorbar(np.arange(1,6)+offset,y,yerr=(bounds[:,1]-bounds[:,0])/2,
                    fmt='o',capsize=5,ms=7,color=color,label=label)
    ax.set(xticks=range(1,6),xlabel='Previously opened trajectory midpoint',
           ylabel='A actions (same number of adjoint actions)',ylim=(115,190))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper center',ncol=3,frameon=False,fontsize=10)
    ax.set_title('No extra warm-start benefit at actual 1% accuracy: 0/5\n'
                 'Both-path intervals; first and sustained hits agree',fontsize=13)
    fig.supxlabel('Retrospective truth-oracle crossings, not a deployable stop or measured speedup',fontsize=10)
    fig.savefig(ROOT/'assets/figures'/f'{STEM}.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
