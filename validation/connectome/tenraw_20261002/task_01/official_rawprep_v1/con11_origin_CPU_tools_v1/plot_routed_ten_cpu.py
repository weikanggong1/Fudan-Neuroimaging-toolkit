#!/usr/bin/env python3
"""Display only actually verified official CPU reference images and separately recorded times."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery-summary',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    summary=json.loads(a.delivery_summary.read_text());subjects=list(summary['cases']);a.output_root.mkdir(parents=True,exist_ok=False)
    n=len(subjects);cols=n if n<=5 else (n+1)//2;rows=(n+cols-1)//cols;fig,axes=plt.subplots(rows,cols,figsize=(3*cols,3.7*rows),squeeze=False);provenance=[];times=[]
    for axis,s in zip(axes.flat,subjects):
        d=Path(summary['cases'][s]['official_case_directory']);rp=d/'report.json';r=json.loads(rp.read_text());v=json.loads((d/'completed_contract_verified.json').read_text())
        assert r['completed'] and v['report_sha256']==sha(rp)==summary['cases'][s]['report_SHA256']
        ip=d/'eddy/data.nii.gz';mp=d/'mask/nodif_brain_mask.nii.gz'
        assert sha(ip)==v['output_sha256']['eddy/data.nii.gz'] and sha(mp)==v['output_sha256']['mask/nodif_brain_mask.nii.gz']
        image=nib.load(ip);maskimage=nib.load(mp);mask=np.asarray(maskimage.dataobj);index=r['selection']['ap_index'];brain=np.asarray(image.dataobj[...,index]);z=brain.shape[2]//2
        assert image.shape[-1]==102 and brain.shape==mask.shape and np.allclose(image.affine,maskimage.affine,rtol=0,atol=1e-5) and np.isfinite(brain).all()
        axis.imshow(brain[:,:,z].T,origin='lower',cmap='gray',vmin=0,vmax=np.percentile(brain[mask>0],99));axis.contour(mask[:,:,z].T,levels=[.5],colors=['#ffbd59'],linewidths=.8);axis.set_title(f'{s} | corrected AP{index}',fontsize=10);axis.set_axis_off()
        provenance.append({'subject':s,'report_SHA256':sha(rp),'verified_contract_SHA256':sha(d/'completed_contract_verified.json'),'EDDY_image_SHA256':sha(ip),'mask_SHA256':sha(mp),'corrected_frame_index':index,'native_axis2_slice':z,'shape':list(image.shape),'display_resampling':False})
        c={x['stage']:x for x in r['commands']};times.append([c['official_EDDY_CPU']['wall_seconds'],c['official_topup']['wall_seconds'],c['official_synthstrip_CPU']['wall_seconds']])
    for axis in list(axes.flat)[n:]:axis.set_axis_off()
    fig.suptitle(f'Official CPU8 EDDY corrected reference b0 | {n} verified raw cases',fontsize=14);fig.text(.5,.015,'Native midpoint plane; orange: own official SynthStrip mask. No equivalence claim.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.92));fig.subplots_adjust(hspace=.3,wspace=.05);fig.savefig(a.output_root/'verified_CPU_EDDY_brain.png',dpi=180);plt.close(fig)
    fig,axis=plt.subplots(figsize=(11,4.5));x=np.arange(n);t=np.asarray(times)
    for k,(label,color) in enumerate([('New CPU8 EDDY command wall','#4278ac'),('Actual TOPUP command wall (see provenance)','#7b9b61'),('Actual SynthStrip command wall (see provenance)','#ffbd59')]):axis.bar(x+(k-1)*.25,t[:,k],width=.25,label=label,color=color)
    axis.set_xticks(x,subjects);axis.set_ylabel('Command wall seconds');axis.set_title('Actual command walls | first 9 restored stages, CON11 fresh');axis.legend(frameon=False,fontsize=9);axis.spines[['top','right']].set_visible(False);fig.text(.5,.015,'First 9: restored stage walls; CON11: fresh stages. No stage sum as total. CPU8; prefix concurrency 2, CON11 1.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(a.output_root/'verified_CPU_command_times.png',dpi=180);plt.close(fig)
    value={'scope':'actual verified official CPU references only; visualization does not alter scientific images','delivery_summary_SHA256':sha(a.delivery_summary),'tool_SHA256':sha(Path(__file__)),'numpy':np.__version__,'nibabel':nib.__version__,'matplotlib':matplotlib.__version__,'case_execution_kinds':{s:summary['cases'][s]['execution_kind'] for s in subjects},'cases':provenance,'timings_columns':['new_CPU8_EDDY','actual_TOPUP_see_case_execution_kind','actual_SynthStrip_see_case_execution_kind'],'timings_seconds':times,'figure_SHA256':{p.name:sha(p) for p in a.output_root.glob('*.png')}}
    (a.output_root/'provenance.json').write_text(json.dumps(value,indent=2)+'\n')
if __name__=='__main__':main()
