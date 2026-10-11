"""冻结同输入curv.stats的平均/持久过滤/label语义；不进入生产输入链。"""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import torch
import fnit.recon_all
from benchmark_recon_full_curvature_stats import sha, map_comparison, stats_comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject', type=Path, required=True)
    p.add_argument('--candidate-directory', type=Path, required=True)
    p.add_argument('--reference-program', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    fnit.recon_all.__path__.insert(0,str(a.candidate_directory.resolve()))
    module=importlib.import_module('fnit.recon_all.curvature_stats_python')
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    label=a.subject/'label/lh.cortex.label'
    presets=[('averages3', {'averages':3}, ['-a','3']),
             ('cortex_label', {'label_file':label}, ['-l',str(label)]),
             ('lowpass', {'low_pass':.25}, ['--lowPassFilter','.25']),
             ('highpass', {'high_pass':.01}, ['--highPassFilter','.01']),
             ('combined', {'averages':3,'label_file':label,'low_pass':.25,'high_pass':.01},
                ['-a','3','-l',str(label),'--lowPassFilter','.25','--highPassFilter','.01'])]
    files=[a.subject/'surf'/f'lh.{s}' for s in ('smoothwm','curv','sulc')]+[label]
    report={'status':'running','hostname':platform.node(),'threads':a.threads,
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'device':a.device,
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
            'script_sha256':sha(__file__),'candidate_sha256':sha(module.__file__),
            'reference_program_sha256':sha(a.reference_program),
            'input_sha256':{f.name:sha(f) for f in files},'cases':[]}
    save=lambda:(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for name,kwargs,flags in presets:
        native=a.output/name/'reference'
        for folder in ('surf','stats','scripts'):(native/folder).mkdir(parents=True)
        for f in files[:-1]:shutil.copyfile(f,native/'surf'/f.name)
        command=[str(a.reference_program.resolve()),'-m','--writeCurvatureFiles','-G',*flags,
          '-o',str(native/'stats/lh.curv.stats'),'-F','smoothwm',native.name,'lh','curv','sulc']
        env=dict(os.environ,SUBJECTS_DIR=str(native.parent.resolve()),FREESURFER_HOME=str(a.assets.resolve()))
        tick=time.perf_counter()
        with (native/'native.log').open('w') as stream:
            subprocess.run(command,env=env,cwd=native/'scripts',stdout=stream,stderr=subprocess.STDOUT,check=True)
        native_seconds=time.perf_counter()-tick
        destination=a.output/name/'candidate'
        result=module.write_curvature_stats(subject_dir=a.subject,hemisphere='lh',stats_file=destination/'lh.curv.stats',
          output_prefix=destination/'lh.smoothwm',device=a.device,angle_backend='source_libm',**kwargs)
        row={'profile':name,'kwargs':{k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()},
            'native_seconds':native_seconds,'candidate':result,
            'maps':{k:map_comparison(native/'surf'/f'lh.smoothwm.{k}.crv',result['outputs'][k]) for k in module.MAP_ORDER},
            'stats':stats_comparison(native/'stats/lh.curv.stats',result['outputs']['stats'])}
        report['cases'].append(row);save()
    report['input_sha256_after']={f.name:sha(f) for f in files}
    report['status']='complete';save()


if __name__=='__main__':main()
