"""公开基准 DVS 状态图；只发布图和汇总，不复制网格或参考字段。"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark_recon_topology_candidate_state import load_state_capture
from fnit.recon_all.topology_candidate_state import record_candidate_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['record','before','restored','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    record,selection,_=load_state_capture(a.record)
    before,_,_=load_state_capture(a.before)
    restored,_,_=load_state_capture(a.restored)
    snapshot=record_candidate_state(surface_state=record,**selection)
    vertices=snapshot.vertex_numbers[snapshot.restore_vertex]
    reference=record['original_vertices'][vertices]
    displacement=np.linalg.norm(before['original_vertices'][vertices].astype(float)-reference,axis=1)
    restored_error=np.linalg.norm(restored['original_vertices'][vertices].astype(float)-reference,axis=1)
    figure,axes=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
    center=reference.mean(axis=0)
    axes[0].scatter(reference[:,0]-center[0],reference[:,1]-center[1],s=15,label='Recorded orig')
    axes[0].scatter(before['original_vertices'][vertices,0]-center[0],
                    before['original_vertices'][vertices,1]-center[1],s=9,label='Before restore')
    axes[0].set(xlabel='Local surface RAS X (mm)',ylabel='Local surface RAS Y (mm)',title=f'{len(vertices)} restored records')
    axes[0].legend(fontsize=7);axes[0].set_aspect('equal','datalim')
    axes[1].plot(np.sort(displacement),label='Before restore')
    axes[1].plot(np.sort(restored_error),label='After source restore')
    axes[1].set(xlabel='Ordered rank',ylabel='Orig displacement (mm)',title='Selected orig state');axes[1].legend(fontsize=7)
    counts=[record['nfaces'],before['nfaces'],restored['nfaces']]
    axes[2].bar(['Record','Candidate','Restored'],np.asarray(counts)-record['nfaces'],color=['#3465a4','#e8ad31','#47966c'])
    axes[2].set(ylabel='Change in logical face count',title='Face capacity is retained')
    for i,value in enumerate(counts):axes[2].text(i,value-record['nfaces']+2,str(value),ha='center',fontsize=8)
    figure.suptitle('Public ds000114 sub06 LH · frozen DVS transaction · no mesh-quality claim',fontsize=11)
    a.output.parent.mkdir(parents=True,exist_ok=True);figure.savefig(a.output,dpi=160);plt.close(figure)


if __name__=='__main__':main()
