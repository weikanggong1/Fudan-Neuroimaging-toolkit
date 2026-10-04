"""Compare saved same-input stage arrays; fixed existing morph gates only."""
import argparse,hashlib,json,pathlib,time,os

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 t=time.perf_counter();a=argparse.ArgumentParser();a.add_argument('--left',required=True,type=pathlib.Path);a.add_argument('--right',required=True,type=pathlib.Path);a.add_argument('--output',required=True,type=pathlib.Path);args=a.parse_args()
 if args.output.exists():raise FileExistsError('new comparison output required')
 import numpy as np
 controllers=[json.loads((p.parent/'diagnostics/controller.json').read_text()) for p in [args.left,args.right]]
 if any(c.get('status')!='complete' or c.get('exit_code')!=0 or not c.get('all_tracked_owned_exited') or c.get('cancellation_signals') for c in controllers):raise ValueError('both arms need completed controller/cleanup receipts')
 reports=[json.loads((p/'report.json').read_text()) for p in [args.left,args.right]]
 if any(r.get('status')!='complete' or len(r.get('passes',[]))!=2 for r in reports):raise ValueError('both real cold/warm arms must complete')
 for key in ['white_input_sha256','pial_input_sha256','vertices','faces']:
  if reports[0]['space'][key]!=reports[1]['space'][key]:raise ValueError('mesh input correspondence mismatch')
 if reports[0]['input_manifest_sha256']!=reports[1]['input_manifest_sha256']:raise ValueError('different input manifests')
 if any(not r['space']['ordered_faces_equal'] for r in reports):raise ValueError('ordered faces must correspond')
 args.output.mkdir(parents=True);read_files={};comparisons=[]
 def read_npz(p):
  read_files[str(p)]=digest(p);return np.load(p,allow_pickle=False)
 def numeric(x,y,tolerance=None):
  if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('numeric shape/finiteness mismatch')
  d=np.abs(x.astype(np.float64)-y.astype(np.float64));z={'shape':list(x.shape),'left_dtype':str(x.dtype),'right_dtype':str(y.dtype),'exact':bool(np.array_equal(x,y)),'nonzero_difference_elements':int(np.count_nonzero(d)),'max_abs':float(d.max()) if d.size else 0,'p99_abs':float(np.quantile(d,.99)) if d.size else 0,'mean_abs':float(d.mean()) if d.size else 0}
  if tolerance:
   atol,rtol=tolerance;bad=d>atol+rtol*np.abs(x.astype(np.float64));z.update(existing_operator_tolerance={'atol':atol,'rtol':rtol},outlier_count=int(np.count_nonzero(bad)),operator_pass=not bool(np.any(bad)))
  else:z['equivalence_assessment']='not_assessed; no new tolerance defined'
  return z
 pairs=[('A8f_cold_vs_warm',args.left/'cold',args.left/'warm'),('B3a_cold_vs_warm',args.right/'cold',args.right/'warm'),('A8f_vs_B3a_cold',args.left/'cold',args.right/'cold'),('A8f_vs_B3a_warm',args.left/'warm',args.right/'warm')]
 for name,left,right in pairs:
  item={'pair':name,'metrics':{},'roi_tables':{}};x=read_npz(left/'metrics.npz');y=read_npz(right/'metrics.npz')
  if set(x.files)!=set(y.files):raise ValueError('metric map coverage differs')
  for metric in x.files:
   tol=(.001,.001) if metric in ['area','area.pial'] else (.005,.001) if metric in ['thickness','curv','curv.pial'] else None
   item['metrics'][metric]=numeric(x[metric],y[metric],tol)
  for atlas in ['aparc','aparc.a2009s','aparc.DKTatlas','aparc.pial','BA_exvivo','BA_exvivo.thresh']:
   x=read_npz(left/(atlas+'.npz'));y=read_npz(right/(atlas+'.npz'));nx=list(x['names']);ny=list(y['names'])
   if len(set(nx))!=len(nx) or len(set(ny))!=len(ny) or set(nx)!=set(ny):raise ValueError('named ROI coverage differs')
   rx=x['values'];ry=y['values'][[ny.index(n) for n in nx]];columns=['NumVert','SurfArea','GrayVol','ThickAvg','ThickStd','MeanCurv','GausCurv','FoldInd','CurvInd'];table={col:numeric(rx[:,i],ry[:,i]) for i,col in enumerate(columns)}
   table['NumVert']['integer_counts_exact']=bool(np.array_equal(rx[:,0],ry[:,0]));table['definition_scope']='3a ROI SurfArea corrects per-face float32 share then float64 sum; no old/new byte-equivalence requirement';item['roi_tables'][atlas]={'rows':len(nx),'names_equal':True,'columns':table}
  comparisons.append(item)
 for path,sha in read_files.items():
  if digest(pathlib.Path(path))!=sha:raise ValueError('saved stage arrays changed during analysis')
 out={'status':'complete','scope':'saved same-input lh metrics+ROI numerical diagnostic only','whole_case_equivalence':'not_assessed','coordinate_space':reports[0]['space'],'input_manifest_sha256':reports[0]['input_manifest_sha256'],'reports':{str(p/'report.json'):digest(p/'report.json') for p in [args.left,args.right]},'read_file_sha256':read_files,'comparisons':comparisons,'timings':{name:[{'pass':v['name'],'entry_seconds_including_output_serialization':v['entry_seconds_including_output_serialization'],'stages':v['stages'],'metrics_internal_seconds':v['metrics_internal_seconds'],'cache_counters':v['cache_counters']} for v in r['passes']] for name,r in zip(['A8f','B3a'],reports)},'cpu_analysis_seconds':time.perf_counter()-t,'performance_assessment':'observations only; single sequential A/B and process warming, shared GPU; no speed or causal attribution claim'}
 (args.output/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
