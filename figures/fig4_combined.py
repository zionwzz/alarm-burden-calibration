#!/usr/bin/env python3
"""Render a single vector figure combining duration and overshoot reward estimates."""
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fig4_reward as src
from alarmreplay import figstyle_lrc as st

def main():
    st.use_style()
    mpl.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,
                         'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42})
    cells,over=src.load_cells(),src.load_overshoot()
    st.UNSUPPORTED = "#75808A"
    chans=st.CHANNEL_ORDER
    fig=plt.figure(figsize=(6.5,6.25))
    outer=fig.add_gridspec(2,2,left=.105,right=.975,top=.953,bottom=.162,wspace=.22,hspace=.39)
    cmap=LinearSegmentedColormap.from_list('reward',['#F6F4F0','#DFA856','#B87520','#6E3405'])
    cmap.set_bad('white')
    images=[]
    for k,ch in enumerate(chans):
        sub=outer[k//2,k%2].subgridspec(2,1,height_ratios=[1.35,1],hspace=.70)
        ax=fig.add_subplot(sub[0]);src.ratio_panel(ax,cells[ch],ch==st.LEAST_STABLE,k%2==0)
        ax.set_title(f'{chr(65+k)}  {src.CHANNEL_LABEL[ch]}',loc='left',fontsize=9,pad=6,fontweight='bold')
        ax.set_yticks([0,.5,1]);ax.set_xticks([0,40,80,120]);ax.tick_params(labelsize=7.5)
        ax.set_xlabel('Duration (s)',labelpad=1,fontsize=7.5)
        # Remove duplicated tail annotations and label bins compactly; all estimates retained.
        for q in list(ax.texts):
            if 'pooled' in q.get_text():q.remove()
        ax.text(.845,-.21,'pooled tail',transform=ax.transAxes,ha='center',fontsize=7.2,color=st.MUTED)
        hm=fig.add_subplot(sub[1])
        groups=[g for g in src.DURATION_GROUPS if g in over[ch]]
        grid=np.full((len(groups),6),np.nan)
        for ri,g in enumerate(groups):
            for z,h,n in over[ch][g]:grid[ri,z]=h
        mesh=hm.pcolormesh(np.arange(7)-.5,np.arange(len(groups)+1)-.5,
                          np.ma.masked_invalid(grid),cmap=cmap,vmin=0,vmax=1.1,
                          shading='flat',rasterized=False)
        reported=np.isfinite(grid).any(axis=0)
        for ri in range(len(groups)):
            for ci in range(6):
                if not np.isfinite(grid[ri,ci]):
                    style=src.BLANK_UNDER_FLOOR if reported[ci] else src.BLANK_NO_CELL
                    hm.add_patch(plt.Rectangle((ci-.5,ri-.5),1,1,**style))
        hm.set_xticks(range(6),['0','1','2','4','8','16+'])
        hm.set_yticks(range(len(groups)),[src.DURATION_TICK[g] for g in groups] if k%2==0 else ['']*len(groups))
        hm.set_xlim(-.5,5.5);hm.set_ylim(-.5,len(groups)-.5)
        hm.set_xlabel('Overshoot (channel steps)',fontsize=7.5,labelpad=2)
        hm.tick_params(labelsize=7.5,length=0);hm.grid(False)
        for spine in hm.spines.values():spine.set_visible(False)
        if k%2==0:hm.set_ylabel('Duration (s)',fontsize=7.5,labelpad=2)
        images.append(mesh)
    cax=fig.add_axes([.15,.058,.32,.016]);cb=fig.colorbar(images[0],cax=cax,orientation='horizontal')
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
    cb.set_ticks([0,.5,1.1]);cb.set_label('Annunciation ratio',fontsize=8,labelpad=1)
    cb.ax.tick_params(labelsize=7.5,length=2);cb.outline.set_visible(False)
    handles=[Line2D([],[],color=st.LRC_D,marker='o',ls='',markersize=3,label='20 or more excursions'),
             Line2D([],[],color=st.LRC_D,marker='o',mfc='white',ls='',markersize=3,label='fewer than 20'),
             Patch(facecolor='white',edgecolor='#AEB3B8',hatch='////',label='ratio not reported'),
             Patch(facecolor='#E7E9EA',edgecolor='#AEB3B8',label='group not reported')]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.53,.064),fontsize=7.5,frameon=False,
               handlelength=1.5,labelspacing=.35)
    out=Path(src.OUT);out.mkdir(exist_ok=True)
    fig.savefig(out/'Figure4_reward.pdf',format='pdf')
    fig.savefig(out/'Figure4_reward.png',dpi=180)
    plt.close(fig)
if __name__=='__main__':main()
