"""原始T1输入链的首次CUDA分配对照；不运行官方参考，不改生产源码。

--config JSON指定input/output/weights/assets/device/threads/prime_child。
output须不存在；原网格/conformed网格、surface RAS约定沿用输入链。
prime_child=False原子进程入口不变；True只在子进程加载CPU权重前创建并删除
同设备float32单元素张量及同步，不改TF32/模型/算法。两种均使用相同缓存策略。
输出diagnostic.json、原输入链影像/LTA/XFM；异常保留错误和traceback并返回1。
记录当前脚本/配置SHA、代码版本、主机及相关内存，不把该诊断称完整整例。
属于子进程内部初始化排错，没有独立官方等价命令。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import torch
from fnit.recon_all.input_talairach_chain import run_input_talairach_chain
from fnit.recon_all.profiling import configure_cuda_allocator, StageProfiler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args(); c=json.loads(args.config.read_text()); out=Path(c["output"])
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(c["threads"])
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    allocator=configure_cuda_allocator(device=c["device"],policy="disabled")
    original_run=subprocess.run
    def run(command,*a,**kw):
        if c["prime_child"] and isinstance(command,list) and "fnit.recon_all.talairach_synthmorph" in command:
            bootstrap=("import runpy,sys,torch,json;torch.set_num_threads("+str(c["threads"])+");"
                       "p=torch.empty(1,device="+repr(c["device"])+");torch.cuda.synchronize("+repr(c["device"])+");del p;"
                       "print(json.dumps({'bootstrap':'explicit first allocation','free_total_bytes':torch.cuda.mem_get_info("+repr(c["device"])+")}),flush=True);"
                       "sys.argv=['fnit.recon_all.talairach_synthmorph',*sys.argv[1:]];"
                       "runpy.run_module('fnit.recon_all.talairach_synthmorph',run_name='__main__')")
            command=[command[0],"-c",bootstrap,*command[3:]]
        return original_run(command,*a,**kw)
    subprocess.run=run
    report={"config":c,"config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "host":socket.gethostname(),"meminfo_before":Path('/proc/meminfo').read_text(),
            "allocator":allocator,"status":"running"}
    profiler=StageProfiler(device=c["device"],synchronize=True,allocator=allocator)
    start=time.perf_counter();code=0
    try:
        report["result"]=profiler.run("input_talairach",run_input_talairach_chain,
            t1=c["input"],subject_dir=out,weights_dir=c["weights"],assets_dir=c["assets"],
            device=c["device"],threads=c["threads"])
        report["status"]="complete"
    except Exception as error:
        report.update(status="failed",error=repr(error),traceback=traceback.format_exc());code=1
    finally:
        subprocess.run=original_run
        report.update(seconds=time.perf_counter()-start,stage=profiler.last_row,
                      meminfo_after=Path('/proc/meminfo').read_text())
        (out/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(code)


if __name__=='__main__':main()
