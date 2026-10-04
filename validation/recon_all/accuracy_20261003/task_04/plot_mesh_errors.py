"""将真实 orig 点到三角面误差绘成三视图；字体由运行环境提供。"""
import argparse,json
from pathlib import Path
import nibabel.freesurfer.io as fsio
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--assessment-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,required=True);p.add_argument('--subject-label',required=True);p.add_argument('--stage',default='orig');a=p.parse_args();font=FontProperties(fname=str(a.font));fig,axes=plt.subplots(2,3,figsize=(12,7));maximum=[];sets=[]
    for hemi in ('lh','rh'):
        cache=a.assessment_root/(hemi+'.'+a.stage+'.candidate_to_reference.npz')
        if not cache.exists():sets.append(None);continue
        vertices,_=fsio.read_geometry(str(a.candidate/'surf'/(hemi+'.'+a.stage)));d=np.load(a.assessment_root/(hemi+'.'+a.stage+'.candidate_to_reference.npz'))['distance_mm'];sets.append((vertices,d));maximum.append(float(np.quantile(d,.99)))
    vmax=max(maximum)
    for row,item in enumerate(sets):
        if item is None:
            for ax in axes[row]:ax.text(.5,.5,'该半球尚待计算',ha='center',va='center',fontproperties=font);ax.axis('off')
            continue
        v,d=item
        order=np.argsort(d,kind='stable')
        for column,(x,y,view) in enumerate(((1,2,'矢状面'),(0,2,'冠状面'),(0,1,'轴位'))):
            ax=axes[row,column];plot=ax.scatter(v[order,x],v[order,y],c=d[order],vmin=0,vmax=vmax,s=.8,cmap='magma',rasterized=True);ax.set_aspect('equal');ax.set_title(('左半球 ' if row==0 else '右半球 ')+view,fontproperties=font);ax.set_xlabel('surface RAS (mm)');ax.set_ylabel('surface RAS (mm)')
    fig.suptitle(a.subject_label+'：'+a.stage+' 到官方三角面的局部距离',fontproperties=font);fig.subplots_adjust(right=.86,hspace=.35,top=.90);cax=fig.add_axes((.9,.15,.02,.65));bar=fig.colorbar(plot,cax=cax);bar.set_label('距离 (mm)，色条上限为已完成侧最大 P99',fontproperties=font);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);plt.close(fig)
    if a.output.suffix.lower()=='.svg':a.output.write_text('\n'.join(line.rstrip() for line in a.output.read_text().splitlines())+'\n')
if __name__=='__main__':main()
