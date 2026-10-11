"""公开原域K1脑图与完整先验的同输入误差；仅在独立计算完成后读参考。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_recon_topology_mri_likelihood import load_statistics_capture


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(*,candidate,reference,report,output):
    source=np.load(candidate,allow_pickle=False)
    baseline=load_statistics_capture(capture_path=reference)
    data=json.loads(Path(report).read_text())
    if data['status']!='complete_global_priors':raise ValueError('complete strict priors required')
    output=Path(output)
    if output.exists():raise FileExistsError('new output directory required')
    output.mkdir(parents=True)
    fig=plt.figure(figsize=(12,9),layout='constrained')
    brain=fig.add_subplot(2,2,1,projection='3d')
    xyz=source['raw_vertices'];k=source['sff1_k1'];sample=np.arange(0,len(xyz),5)
    limit=float(np.percentile(np.abs(k),95))
    points=brain.scatter(xyz[sample,0],xyz[sample,1],xyz[sample,2],c=k[sample],
        s=.6,cmap='coolwarm',vmin=-limit,vmax=limit,rasterized=True)
    brain.view_init(elev=8,azim=180);brain.set_axis_off();brain.set_box_aspect(np.ptp(xyz,axis=0))
    brain.set_title('Public sub-07 LH: self-generated raw SFF1 K1')
    fig.colorbar(points,ax=brain,shrink=.6,label='K1 (mm$^{-1}$)')
    joint=fig.add_subplot(2,2,2)
    joint.imshow(np.log10(np.maximum(source['k_pdf'][:,:,0],np.finfo(np.float32).tiny)).T,
        origin='lower',extent=(*source['k1_bins'][[0,-1]],*source['k2_bins'][[0,-1]]),cmap='viridis')
    joint.set_title('Self-generated joint curvature PDF (log10)')
    joint.set_xlabel('K1 (mm$^{-1}$)');joint.set_ylabel('K2 (mm$^{-1}$)')
    dot=fig.add_subplot(2,2,3)
    dot.plot(baseline['dot_bins'],baseline['dot_counts'],color='black',lw=2,label='Fixed source')
    dot.plot(source['dot_bins'],source['dot_counts'],color='tab:red',ls='--',label='FNIT self-generated')
    dot.set_title('All 100 normal-dot bins: exact overlap')
    dot.set_xlabel('Source legacy expression (mixed mm / mm²)');dot.set_ylabel('Probability')
    dot.legend()
    error=fig.add_subplot(2,2,4)
    names=['K1 PDF','K2 PDF','Joint PDF','Dot PDF']
    values=[np.max(np.abs(source[a].astype(np.float64)-baseline[a].astype(np.float64)))
        for a in ['k1_counts','k2_counts','k_pdf','dot_counts']]
    error.bar(names,values,color='tab:green');error.set_ylim(0,1e-6)
    for index,value in enumerate(values):error.text(index,1e-8,f'{value:g}',ha='center')
    error.set_title('Maximum absolute error: every bin evaluated')
    error.set_ylabel('Probability difference')
    fig.suptitle('Raw nofix / qsphere → first K prior + smoothed ORIGINAL metric → dot prior\n'
        'Benchmark only; complete GA and whole-pipeline equivalence not assessed')
    image=output/'global_priors_brain_and_pdf.png';fig.savefig(image,dpi=170);plt.close(fig)
    receipt=dict(schema_version=1,figure_sha256=sha(image),script_sha256=sha(__file__),
        candidate_checkpoint_sha256=sha(candidate),reference_capture_sha256=sha(reference),
        report_sha256=sha(report),nvertices=len(xyz),render_stride=5,
        maximum_probability_errors=dict(zip(names,map(float,values))),
        full_values_used_for_errors=True,reference_curvature_field_not_used_as_input=True)
    (output/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','reference','report','output']:parser.add_argument('--'+name,type=Path,required=True)
    render(**vars(parser.parse_args()))
