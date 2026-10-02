"""一次最小首次CUDA分配探针；保留错误和环境元数据，不修改分配器或精度。

--output 指定JSON；device默认cuda:0。需外层先绑定UUID/线程并取得共用锁。
只保存明确CUDA/PyTorch键、库版本、GPU查询和meminfo，不保存全量环境。
"""
import argparse,json,os,subprocess,traceback
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(4)
    keys=['CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','CUDA_MODULE_LOADING','CUDA_HOME',
          'PYTORCH_NO_CUDA_MEMORY_CACHING','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF',
          'CUDA_LAUNCH_BLOCKING','LD_LIBRARY_PATH','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']
    report={'pid':os.getpid(),'torch_file':torch.__file__,'torch_version':torch.__version__,
      'torch_cuda_version':torch.version.cuda,'device':a.device,'environment':{k:os.environ.get(k) for k in keys},
      'meminfo_before':Path('/proc/meminfo').read_text(),'cuda_initialized_before':torch.cuda.is_initialized()}
    query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free','--format=csv'],capture_output=True,text=True)
    report['nvidia_smi']={'returncode':query.returncode,'stdout':query.stdout,'stderr':query.stderr}
    code=0
    try:
        tensor=torch.empty(1,dtype=torch.float32,device=a.device)
        torch.cuda.synchronize(a.device)
        report.update(status='passed',free_total_bytes=torch.cuda.mem_get_info(a.device),
          gpu_uuid=str(torch.cuda.get_device_properties(a.device).uuid),allocated_bytes=tensor.numel()*tensor.element_size())
    except Exception as error:
        code=1;report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    report.update(returncode=code,meminfo_after=Path('/proc/meminfo').read_text(),cuda_initialized_after=torch.cuda.is_initialized())
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    raise SystemExit(code)
if __name__=='__main__':main()
