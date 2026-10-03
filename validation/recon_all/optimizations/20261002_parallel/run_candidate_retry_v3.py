"""等待完整 CUDA 初始化对照完成，再显式启动一次原始T1空目录整例复测。

输入 --round、已安装Conda --python、共用 --lock。输出新的 retry_v3 状态与
队列；仅八批诊断全部初始化成功且原配置/source/原生资源仍匹配才启动。失败明确停止，
无自动重试；排队/初始化诊断不计入重建墙钟，旧失败目录完整保留。
"""
import argparse,datetime,hashlib,json,pathlib,subprocess,time
P=pathlib.Path

def write(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--round',type=P,required=True);a.add_argument('--python',type=P,required=True);a.add_argument('--lock',type=P,required=True);args=a.parse_args();p=args.round;c=p/'coordinator';status=c/'candidate_retry_v3_status.json';report={'status':'waiting_for_cuda_diagnostic','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':digest(P(__file__))};write(status,report)
 try:
  deadline=time.monotonic()+8*3600
  while True:
   diagnostic=json.loads((p/'cuda_bootstrap_pairs_v2/summary.json').read_text()) if (p/'cuda_bootstrap_pairs_v2/summary.json').exists() else {}
   if diagnostic.get('status')=='complete':
    if len(diagnostic['batches'])!=8 or not all(row['both_initialized'] for row in diagnostic['batches']):raise RuntimeError('paired cold-start diagnosis contains failed batches; do not launch')
    break
   if diagnostic.get('status')=='failed':raise RuntimeError('cold-start diagnostic failed; do not launch whole case')
   if time.monotonic()>deadline:raise TimeoutError('diagnostic did not complete within 8h')
   time.sleep(15)
  old=c/'candidate_sub02_8d750e2.json';new=c/'candidate_sub02_8d750e2_retry_v3.json';previous=json.loads(old.read_text());cfg=json.loads(new.read_text())
  assert digest(old)=='8be09eb143fdf8a2526d1983f2f7df08678a1c463a7c90b0297be28cdbe8195f'
  assert set(k for k in cfg if cfg[k]!=previous[k])=={'output','diagnostic_root'}
  assert cfg['code_commit']=='8d750e25d4d067a43edb788a96b2086a1c031ba0'
  assert not P(cfg['output']).exists() and not P(cfg['diagnostic_root']).exists()
  proof=json.loads((p/'candidate_install_8d750e2/installed-source-check.json').read_text());assert proof['status']=='passed' and proof['mismatches']==[]
  installed=(P(cfg['code_root'])/'src').resolve();assert installed==P(proof['installed_import_root']).resolve()
  for filename,expected in proof['source_sha256'].items():assert digest(installed/'fnit/recon_all'/filename)==expected
  native=json.loads((p/'private_install_v1/installed-native-optimizations.json').read_text())
  for row in native['programs'].values():assert digest(P(row['path']))==row['sha256']
  report.update(status='whole_queue',diagnostic_sha256=digest(p/'cuda_bootstrap_pairs_v2/summary.json'),config_sha256=digest(new),runtime_code_commit=cfg['code_commit']);write(status,report)
  argv=[str(args.python),str(c/'run_whole_queue.py'),'--configs',str(new),'--lock',str(args.lock),'--output',str(c/'candidate_retry_queue_v2'),'--monitor',str(c/'run_monitored.py'),'--launcher',str(c/'execute_whole_case.py')]
  report['command']=argv;write(status,report)
  with (c/'candidate_retry_queue_v2.log').open('x') as log:code=subprocess.call(argv,stdout=log,stderr=subprocess.STDOUT)
  report.update(status='complete' if code==0 else 'failed',exit_code=code)
 except BaseException as e:report.update(status='failed',error=repr(e),exit_code=1);raise
 finally:report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(status,report)
 return report['exit_code']
if __name__=='__main__':raise SystemExit(main())
