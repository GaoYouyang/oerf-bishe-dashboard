"""Render the five retrospective TRAIN gradient cosines, not speedup."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_solver_loss_alignment_20260908'
d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
fig,ax = plt.subplots(figsize=(8,3.8),layout='constrained')
ax.bar(range(1,6),d['cosines'],color='#287a72',width=.55)
ax.axhline(0,color='#444444',linewidth=1)
ax.set(ylim=(-.12,1),xticks=range(1,6),xlabel='TRAIN fold (overlapping pairs)',ylabel='Gradient cosine',
    title='No local descent conflict across five fixed TRAIN folds')
for i,value in enumerate(d['cosines'],1):
    ax.text(i,value+.02,f'{value:.3f}',ha='center',va='bottom')
ax.text(.98,.96,'Not outer testing or a speedup result',transform=ax.transAxes,ha='right',va='top',fontsize=9,color='#555555')
ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png',dpi=170)
plt.close(fig)
