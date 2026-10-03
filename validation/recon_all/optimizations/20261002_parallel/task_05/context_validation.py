"""显式CUDA初始化后在同一进程验证最新warp、完整GPU/Conda阶段。

--config沿用stage配置；output必须新目录；--expected-gpu-uuid强校验实际GPU。
一次init/set-device/properties/memory-info/4B分配，不自动重试。成功才跑
14项单元、两例算子、完整阶段AB/BA；未清空文件/JIT缓存，逐次重建模型。
外层须持有共享锁；allocator由启动环境显式选择，不在程序内更改。
"""
import argparse,hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--expected-gpu-uuid',required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=4:torch.set_num_interop_threads(4)
    record={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'code_commit':cfg['code_commit'],
      'pid':os.getpid(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'validation_started':False,
      'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTORCH_NO_CUDA_MEMORY_CACHING','CUDA_DEVICE_ORDER','CUDA_MODULE_LOADING']},
      'meminfo':Path('/proc/meminfo').read_text(),'single_bootstrap_attempt':True}
    q=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used,memory.free','--format=csv'],capture_output=True,text=True)
    record['nvidia_before']={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
    try:
        record['attempt_stage']='cuda_init';torch.cuda.init()
        record['attempt_stage']='set_device';torch.cuda.set_device(0)
        record['attempt_stage']='properties';props=torch.cuda.get_device_properties(0);record['actual_gpu_uuid']=str(props.uuid)
        if str(props.uuid).lower().removeprefix('gpu-')!=a.expected_gpu_uuid.lower().removeprefix('gpu-'):raise RuntimeError('actual CUDA UUID differs from declared target')
        record['attempt_stage']='memory_info';record['free_total_before']=torch.cuda.mem_get_info(0)
        record['attempt_stage']='allocation';prime=torch.empty(1,device='cuda:0',dtype=torch.float32)
        record['attempt_stage']='synchronize';torch.cuda.synchronize(0);del prime
        record['status']='passed';record['validation_started']=True
    except Exception as error:
        record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        (out/'bootstrap.json').write_text(json.dumps(record,indent=2)+'\n');raise
    (out/'bootstrap.json').write_text(json.dumps(record,indent=2)+'\n')
    import pytest
    from contextlib import redirect_stdout,redirect_stderr
    root=Path(__file__).resolve().parents[5]
    with (out/'unit.log').open('w') as stream,redirect_stdout(stream),redirect_stderr(stream):
        code=pytest.main(['-q',str(root/'tests/recon_all/test_mni_warp_gpu.py')])
    if code:raise RuntimeError(f'unit regression failed, exit {code}')
    import benchmark,stage_benchmark
    operator=dict(cfg,output=str(out/'operators'));path=out/'operators.config.json';path.write_text(json.dumps(operator,indent=2)+'\n')
    sys.argv=['benchmark.py','--config',str(path),'--mode','operators','--prime-cuda'];benchmark.main()
    pairs=[]
    for index,case in enumerate(cfg['cases']):
        order=['stage','stage-conda'] if index%2==0 else ['stage-conda','stage']
        for mode in order:
            # 同一已验证context；释放无消费者的allocator缓冲，不改变运算精度。
            torch.cuda.empty_cache()
            config=dict(cfg,cases=[case],output=str(out/(case['id']+'_'+mode)))
            path=out/(case['id']+'_'+mode+'.config.json');path.write_text(json.dumps(config,indent=2)+'\n')
            sys.argv=['stage_benchmark.py','--config',str(path),'--mode',mode];stage_benchmark.main()
            report=json.loads((Path(config['output'])/'report.json').read_text());pairs.append({'case':case['id'],'mode':mode,'report':report})
            (out/'stage_pairs.json').write_text(json.dumps({'code_commit':cfg['code_commit'],'bootstrap_strategy':'same initialized context, no model retry','case_order':'sub01 GPU→Conda; sub02 Conda→GPU','pairs':pairs},indent=2)+'\n')
if __name__=='__main__':main()
