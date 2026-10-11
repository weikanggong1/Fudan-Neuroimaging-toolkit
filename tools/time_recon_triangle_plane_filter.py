"""记录完整子进程墙钟；包括解释器启动、最终报告写入和退出。"""
import argparse,json,subprocess,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wall-report',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command[:1]==['--']:command=command[1:]
    if not command:parser.error('a child command is required')
    args.wall_report.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();result=subprocess.run(command,check=False)
    seconds=time.perf_counter()-started
    args.wall_report.write_text(json.dumps({'external_process_wall_seconds':seconds,'exit_code':result.returncode,'boundary':'subprocess start through child exit; includes interpreter startup, imports, compute, final JSON write and exit; excludes wrapper import and own report write'},indent=2)+'\n')
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
