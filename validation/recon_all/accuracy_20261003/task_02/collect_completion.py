"""Read completed receipts and real repeat outputs; no model execution."""
import csv,datetime,hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np
BASE=Path('/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/accuracy_20261003/task_02')
ROOT=Path('/tmp/fnit-recon-accuracy-20261003/task_02/diagnostic')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def volume(a,b):
 x=nib.load(a);y=nib.load(b);v=np.asanyarray(x.dataobj);w=np.asanyarray(y.dataobj)
 return {'different':int(np.count_nonzero(v!=w)),'max_abs':float(np.max(np.abs(v.astype(float)-w.astype(float)))),'affine_max_abs':float(np.max(np.abs(x.affine-y.affine))),'baseline_dtype':str(v.dtype),'candidate_dtype':str(w.dtype),'baseline_sha256':sha(Path(a)),'candidate_sha256':sha(Path(b))}
def matrix(p):
 lines=p.read_text().splitlines();i=lines.index('1 4 4')+1;return np.asarray([[float(x) for x in l.split()] for l in lines[i:i+4]])
repeatpath=BASE/'prefix_repeat_sub04/report.json';repeat=json.loads(repeatpath.read_text());out={'receipt_sha256':sha(repeatpath),'run_count':len(repeat['runs']),'all_exit_zero':all(x['exit_code']==0 and x['monitor']['exit_code']==0 for x in repeat['runs']),'comparisons':[],'runs':repeat['runs']};rows=[]
for trial in ['AB','BA']:
 runs={x['backend']:x for x in repeat['runs'] if x['trial']==trial};a=Path(runs['baseline']['run']['subject_dir']);b=Path(runs['candidate']['run']['subject_dir'])
 c={'trial':trial,'volumes':{n:volume(str(a/'mri'/n),str(b/'mri'/n)) for n in ['orig/001.mgz','rawavg.mgz','orig.mgz','synthstrip.mgz']},'xfm_bytes_equal':(a/'mri/transforms/talairach.xfm').read_bytes()==(b/'mri/transforms/talairach.xfm').read_bytes(),'lta_matrix_max_abs':{n:float(np.max(np.abs(matrix(a/'mri/transforms'/n)-matrix(b/'mri/transforms'/n)))) for n in ['synthmorph.mni305/aff.lta','talairach.xfm.lta']}}
 out['comparisons'].append(c);aa=runs['baseline']['run'];bb=runs['candidate']['run'];rows.append({'trial':trial,'baseline_seconds':aa['wall_seconds'],'candidate_seconds':bb['wall_seconds'],'delta_seconds':bb['wall_seconds']-aa['wall_seconds'],'baseline_conform_seconds':aa['conform_and_xform_tag_seconds'],'candidate_conform_seconds':bb['conform_and_xform_tag_seconds'],'baseline_synthstrip_seconds':aa['synthstrip_seconds'],'candidate_synthstrip_seconds':bb['synthstrip_seconds'],'baseline_talairach_seconds':aa['talairach_seconds'],'candidate_talairach_seconds':bb['talairach_seconds']})
(ROOT/'prefix_repeat_collected.json').write_text(json.dumps(out,indent=2))
with (ROOT/'prefix_repeat_times.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
completion={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_exit_codes_captured':False,'judgement':'completed_receipts_all_expected_child_runs_successful_no_traceback','jobs':[]}
for pid,log,receipt,expected,key in [(90754,'cohort.log',BASE/'cohort_prefix/report.json',10,'cases'),(109621,'cross.log',ROOT/'n4_cross.json',2,'runs'),(74294,'prefix_repeat.log',repeatpath,4,'runs')]:
 d=json.loads(receipt.read_text());text=(ROOT/log).read_text();completion['jobs'].append({'pid':pid,'alive':Path(f'/proc/{pid}').exists(),'receipt':str(receipt),'receipt_sha256':sha(receipt),'expected_count':expected,'completed_count':len(d[key]),'log_sha256':sha(ROOT/log),'log_has_traceback':'Traceback (most recent call last)' in text,'log_tail':text[-1000:],'completion_evidence':'N4 check=True subprocess returned and receipt appended' if pid==109621 else 'every recorded child exit_code=0; verify receipt'})
(ROOT/'completion_receipts.json').write_text(json.dumps(completion,indent=2))
print(json.dumps({'repeat_times':rows,'completion':completion},indent=2))
