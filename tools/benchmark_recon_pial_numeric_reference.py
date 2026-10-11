"""七个冻结自产输入的独立Conda/probe参考；不要求伪造aparc注释。"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import nibabel as nib
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def difference(first,second):
    a,b=np.asarray(first),np.asarray(second)
    if a.shape!=b.shape:return dict(same_shape=False)
    d=np.asarray(a,np.float64)-np.asarray(b,np.float64)
    return dict(same_shape=True,different_elements=int(np.count_nonzero(a!=b)),max_absolute=float(np.abs(d).max()),rms=float(np.sqrt(np.mean(d*d))))
def native_state(path,count):
    dtype=np.dtype([('floats','<f4',(9,)),('flags','<i4',(3,))]);value=np.fromfile(path,dtype=dtype)
    if len(value)!=count:raise ValueError('probe state count differs')
    return value

def publish(path,data):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--hemisphere',choices=['lh','rh'],required=True)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--reference-license',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();h=a.hemisphere
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    names=[f'surf/{h}.white',f'surf/autodet.gw.stats.{h}.dat',f'label/{h}.cortex.label',f'label/{h}.cortex+hipamyg.label',*[f'mri/{x}.mgz' for x in ('brain.finalsurfs','wm','aseg.presurf')]]
    report=dict(scope='isolated seven-input original-source Conda references; no production outputs consumed',status='running',input_sha256={n:sha(a.case/'candidate'/n) for n in names},
        binary_sha256=sha(a.binary),probe_sha256=sha(a.probe),script_sha256=sha(__file__),threads=a.threads,reference_runs=[],operator_threshold='not_assessed',official_installed_reference='not_available_in_current_resource_manifest')
    path=a.output/'report.json';publish(path,report)
    for label,binary in [('conda',a.binary),('probe',a.probe)]:
        root=a.output/label
        for name in names:
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.case/'candidate'/name,target)
        surf,mri,labels=root/'surf',root/'mri',root/'label'
        command=[str(binary),'--adgws-in',str(surf/f'autodet.gw.stats.{h}.dat'),'--seg',str(mri/'aseg.presurf.mgz'),'--threads',str(a.threads),'--wm',str(mri/'wm.mgz'),'--invol',str(mri/'brain.finalsurfs.mgz'),f'--{h}','--i',str(surf/f'{h}.white'),'--o',str(surf/f'{h}.pial.T1'),'--pial','--nsmooth','0','--rip-label',str(labels/f'{h}.cortex+hipamyg.label'),'--pin-medial-wall',str(labels/f'{h}.cortex.label'),'--repulse-surf',str(surf/f'{h}.white'),'--white-surf',str(surf/f'{h}.white'),'--restore-255']
        env=dict(os.environ,FREESURFER_HOME=str(a.assets),SUBJECTS_DIR=str(root.parent),FS_LICENSE=str(a.reference_license))
        if label=='probe':env.update(PLACE_GRAD_PREFIX=str(a.output/'native_gradient'),PLACE_VALUES_PREFIX=str(a.output/'native_targets'))
        tick=time.perf_counter()
        with (a.output/f'{label}.private.log').open('w') as log:child=subprocess.run(command,cwd=mri,env=env,stdout=log,stderr=subprocess.STDOUT)
        report['reference_runs'].append(dict(kind=label,exit_code=child.returncode,full_command_wall_seconds=time.perf_counter()-tick));publish(path,report)
        if child.returncode:raise RuntimeError('independent numeric reference failed')
    c,f=nib.freesurfer.read_geometry(str(a.output/f'conda/surf/{h}.pial.T1'));d,g=nib.freesurfer.read_geometry(str(a.output/f'probe/surf/{h}.pial.T1'))
    report['probe_admission']=dict(coordinates=difference(c,d),ordered_faces_exact=bool(np.array_equal(f,g)))
    publish(path,report)
    if not np.array_equal(c,d) or not np.array_equal(f,g):raise RuntimeError('probe final geometry differs from uninstrumented original-source reference')
    observation=a.case/'pial-observation';events=json.loads((observation/'events.json').read_text());steps=[e for e in events if e['kind']=='step'];comparison=[]
    for event in steps:
        step=event['step'];native=a.output/f'native_gradient.step{step:02d}.after_collision'
        if not native.exists():comparison.append(dict(step=step,status='probe_capture_unavailable'));continue
        state=native_state(native,len(c));clear=native_state(a.output/f'native_gradient.step{step:02d}.clear',len(c))
        targets=np.fromfile(a.output/f'native_gradient.step{step:02d}.intensity_input',dtype='<f4').reshape(-1,2)
        with np.load(observation/event['file']) as py:
            trial_count=len(event['diagnostics'].get('trials',[]))
            row=dict(step=step,pass_index=event['pass_index'],final_trial_candidate_coordinates=difference(state['floats'][:,:3],py[f'trial{trial_count-1}_candidate']),
                     start_coordinates=difference(clear['floats'][:,:3],py['intensity_start']),ripped=difference(clear['flags'][:,0],py['ripped']),
                     actual_target_values=difference(targets[:,0],py['target_values']),target_sigma=difference(targets[:,1],py['target_sigma']),
                     normals=difference(clear['floats'][:,3:6],py['normals']))
            intensity=native_state(a.output/f'native_gradient.step{step:02d}.intensity',len(c))
            tangent=native_state(a.output/f'native_gradient.step{step:02d}.tangential_spring',len(c))
            row['intensity_gradient']=difference(intensity['floats'][:,6:9],py['intensity_gradient'])
            row['complete_gradient']=difference(tangent['floats'][:,6:9],py['trial0_gradient'])
            next_clear=a.output/f'native_gradient.step{step+1:02d}.clear'
            if next_clear.exists():
                row['accepted_coordinates']=difference(native_state(next_clear,len(c))['floats'][:,:3],py['accepted'])
                row['accepted_reference_semantics']='next_iteration_start_coordinates'
            else:row['accepted_reference_semantics']='not_assessed_no_next_clear_capture'
        comparison.append(row);report['step_comparison']=comparison;publish(path,report)
    fields=['start_coordinates','ripped','normals','actual_target_values','target_sigma','intensity_gradient','complete_gradient','final_trial_candidate_coordinates','accepted_coordinates']
    report['first_difference']={field:next((row['step'] for row in comparison if row.get(field,{}).get('different_elements',0)>0),None) for field in fields}
    report['status']='complete_reference_and_admitted_probe'
    report['input_unchanged']=report['input_sha256']=={n:sha(a.case/'candidate'/n) for n in names}
    publish(path,report)
if __name__=='__main__':main()
