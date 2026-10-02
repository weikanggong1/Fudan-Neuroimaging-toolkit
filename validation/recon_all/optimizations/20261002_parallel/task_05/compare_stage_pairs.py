"""比较两例自产完整 GPU/Conda 阶段；仅读已有输出，不运行模型。

--pairs 为 context_validation.py 的 stage_pairs.json；--output 为新 JSON。
CPU 单线程诊断，不作为性能计时。图像、空间、dtype、intent、单位与文件
SHA 分开记录；将新 GPU 与新 Conda 直接比较，不以冻结结果间接推断。
"""
import argparse,hashlib,json
from pathlib import Path
import nibabel as nib
import numpy as np
from benchmark import image_comparison,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();source=json.loads(a.pairs.read_text());group={}
    for entry in source['pairs']:
        group.setdefault(entry['case'],{})[entry['mode']]=entry['report']['cases'][entry['case']]
    result={'code_commit':source['code_commit'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'source_report_sha256':hashlib.sha256(a.pairs.read_bytes()).hexdigest(),'case_order':source['case_order'],
      'overall_equivalence':'not_assessed','strict_138':'unchanged; not rerun',
      'timing_scope':'stage functions include model construction, HDF5, two FP32 forwards and postprocess I/O; comparisons excluded',
      'torch_peak_scope':'cumulative since command CUDA initialization; not independent stage peaks',
      'cache_scope':'same initialized context; fresh models; filesystem/JIT caches not cleared; explicit empty_cache between stages',
      'cases':{}}
    for ident,modes in group.items():
        gpu,cpu=modes['stage'],modes['stage-conda']
        row={'gpu_stage_seconds':gpu['stage_wall_seconds'],'conda_stage_seconds':cpu['stage_wall_seconds'],
          'conda_over_gpu_speedup':cpu['stage_wall_seconds']/gpu['stage_wall_seconds'],
          'gpu_step_seconds':gpu['timings'],'conda_step_seconds':cpu['timings'],'comparisons':{}}
        for name in ('forward','inverse','check'):
            row['comparisons'][name]=image_comparison(gpu['result'][name],cpu['result'][name])
        def deform(item):
            return Path(item['result']['forward']).parent/'tmp/deform.mgz'
        left,right=deform(gpu),deform(cpu)
        x,y=nib.load(str(left)),nib.load(str(right))
        xd,yd=np.asarray(x.dataobj),np.asarray(y.dataobj)
        if xd.shape!=yd.shape:raise ValueError('self-produced deform shapes differ')
        diff=np.abs(xd.astype(np.float64)-yd.astype(np.float64))
        row['comparisons']['self_deform']={'shape':list(xd.shape),'elements':int(diff.size),
          'different_elements':int(np.count_nonzero(diff)),'maximum':float(diff.max()),'p99':float(np.quantile(diff,.99)),
          'affine_maximum':float(np.abs(x.affine-y.affine).max()),'dtype_left':str(xd.dtype),'dtype_right':str(yd.dtype),
          'intent_left':None,'intent_right':None,'units_left':None,'units_right':None,
          'intent_units_scope':'not applicable to MGH header; scanner RAS transform uses mm',
          'sha256_left':sha(left),'sha256_right':sha(right)}
        del xd,yd,diff,x,y
        row['strict_numeric_passed']=all(c['different_elements']==0 for c in row['comparisons'].values())
        row['strict_geometry_passed']=all(c['affine_maximum']==0 for c in row['comparisons'].values())
        row['dtype_intent_units_passed']=all(c['dtype_left']==c['dtype_right'] and c['intent_left']==c['intent_right'] and c['units_left']==c['units_right'] for c in row['comparisons'].values())
        row['internal_forward_contract_passed']={name:item.get('internal_forward_contract_passed','not_observed_in_this_snapshot') for name,item in modes.items()}
        result['cases'][ident]=row
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in d.items() if n!='comparisons'} for k,d in result['cases'].items()},indent=2))
if __name__=='__main__':main()
