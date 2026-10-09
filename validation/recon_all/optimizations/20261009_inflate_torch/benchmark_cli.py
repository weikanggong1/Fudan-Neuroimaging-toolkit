"""冻结v2同输入配对完成后测独立冷CLI，保留空JIT缓存成本。

--source-root 是独立候选源码根，--data 为公开表面manifest，--pair 为
已完成v2报告/参考目录，--output必须不存在。首个child使用新空Numba/
Triton cache，后续复用该缓存，但每次均是新Python进程；CPU父进程不
初始化CUDA。不把CLI/API边界或缓存冷/热混称整例提速。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_complete import compare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    pair=json.loads((args.pair/'summary.json').read_text())
    if pair['status']!='complete_stage_pair':raise ValueError('v2 stage pair must be complete')
    args.output.mkdir(parents=True)
    environment=os.environ.copy()
    environment.update({'PYTHONPATH':str(args.source_root/'src'),
        'NUMBA_CACHE_DIR':str(args.output/'new_numba_cache'),
        'TRITON_CACHE_DIR':str(args.output/'new_triton_cache')})
    data=json.loads((args.data/'manifest.json').read_text())
    report={'scope':'four_real_mesh_cold_python_CLI_not_recon_all','device':args.device,
        'threads':args.threads,'empty_jit_cache_on_first_child':True,'subsequent_children_share_jit_cache':True,
        'source_sha256':{str(Path(__file__)):sha(__file__)},'pair_summary_sha256':sha(args.pair/'summary.json'),
        'status':'running','rows':[],'whole_recon_all_acceleration':'not measured','production_default_changed':False}
    def save():
        p=args.output/'summary.json';p.write_text(json.dumps(report,indent=2)+'\n')
    import sys,torch
    report['parent_cuda_initialized']=torch.cuda.is_initialized()
    try:
        save()
        for index,entry in enumerate(data['cases']):
            source=args.data/entry['surface']
            if sha(source)!=entry['sha256']:raise ValueError('input hash changed')
            directory=args.output/entry['case']/entry['hemisphere'];directory.mkdir(parents=True)
            hemi=entry['hemisphere']
            command=[sys.executable,'-m','fnit.recon_all.inflate_standard_run','--input-surface',str(source),
                '--inflated-output',str(directory/(hemi+'.inflated')),'--sulc-output',str(directory/(hemi+'.sulc')),
                '--backend','torch','--device',args.device,'--threads',str(args.threads),'--report',str(directory/'cli.json')]
            started=time.perf_counter()
            with (directory/'cli.log').open('w') as log:
                result=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
            row={'case':entry['case'],'hemisphere':hemi,'command':command,'wall_seconds':time.perf_counter()-started,
                 'returncode':result.returncode,'jit_cache_scope':'empty on first child' if index==0 else 'reuse shared cache'}
            report['rows'].append(row);save()
            if result.returncode:raise RuntimeError('cold CLI failed; see log')
            row['api']=json.loads((directory/'cli.json').read_text())
            row['to_native']=compare(directory,args.pair/entry['case']/hemi/'native_1',hemi)
            row['to_gpu_v2']=compare(directory,args.pair/entry['case']/hemi/'torch_2',hemi)
            row['inflated_sha256']=sha(directory/(hemi+'.inflated'));row['sulc_sha256']=sha(directory/(hemi+'.sulc'))
            save();print('DONE',entry['case'],hemi,row['wall_seconds'],flush=True)
        report['status']='complete_cold_cli_pair'
    except Exception as error:
        report['status']='failed';report['error']=repr(error);save();raise
    save()


if __name__=='__main__':main()
