"""Only classical reference adequacy, not learned generalization."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = 'poolfire_camera_subset_reference_20260908'
d = json.loads((ROOT/'docs'/f'{STEM}.json').read_text())
rows = sorted(d['camera_counts'],key=lambda r:r['camera_count'])
fig,ax = plt.subplots(figsize=(8,3.8),layout='constrained')
ax.bar([str(r['camera_count']) for r in rows],[r['passing'] for r in rows],color=['#287a72','#a26231'],width=.45)
ax.axhline(505,color='#555555',linestyle=':',linewidth=1)
ax.set(ylim=(0,650),xlabel='Active cameras: two predefined subsets',ylabel='Frames passing all four 1% gates',
    title='Classical references pass with five and seven cameras')
for i,row in enumerate(rows):
    ax.text(i,520,f'{row["passing"]}/505; 5/5 trajectories',ha='center',va='bottom')
ax.text(.98,.98,'Same 505 frames reused; no learned parameters',transform=ax.transAxes,ha='right',va='top',fontsize=9,color='#555555')
ax.spines[['top','right']].set_visible(False)
fig.savefig(ROOT/'assets/figures'/f'{STEM}.png',dpi=170)
plt.close(fig)
