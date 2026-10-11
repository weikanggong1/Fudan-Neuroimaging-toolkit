"""冻结真实首候选门→完整CPU/GPU第一缺陷配对；监测同刻进程树显存。"""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    for k in ['workspace','output','input-surface','input-qsphere','brain']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--gate-only',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    records=[];samples=[];failed=0;peak=None
    stages=[('first','gpu','first-gate')]if args.gate_only else[('first','gpu','first-gate'),('full','cpu','cpu'),('full','gpu','gpu')]
    for mode,backend,name in stages:
        command=[sys.executable,str(args.workspace/'tools/benchmark_recon_topology_gpu_first_ga.py'),'--mode',mode,'--backend',backend,'--device',args.device,'--threads','4','--manifest',str(args.workspace/'SOURCE_MANIFEST.private.json'),'--output',str(args.output/name)]
        for k in ['input_surface','input_qsphere','brain']:command.extend(['--'+k.replace('_','-'),str(getattr(args,k))])
        start=time.perf_counter()
        with (args.output/(name+'.stdout.private.log')).open('w')as out:
            child=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT)
            while child.poll()is None:
                tick=time.monotonic();tree={child.pid,os.getpid()}
                try:
                    parents={}
                    for folder in Path('/proc').iterdir():
                        if not folder.name.isdigit():continue
                        try:
                            status=(folder/'status').read_text()
                            parents[int(folder.name)]=int(next(line.split()[1]for line in status.splitlines()if line.startswith('PPid:')))
                        except (OSError,StopIteration,ValueError):continue
                    while True:
                        expanded=tree|{pid for pid,ppid in parents.items()if ppid in tree}
                        if expanded==tree:break
                        tree=expanded
                    host_tree=set()
                    mapping={}
                    for pid in tree:
                        try:
                            text=(Path('/proc')/str(pid)/'sched').read_text().splitlines()[0]
                            host=int(re.search(r'\((\d+), #threads:',text).group(1));mapping[pid]=host;host_tree.add(host)
                        except (OSError,IndexError,AttributeError,ValueError):pass
                    device_file=args.output/name/'DEVICE.private.json'
                    selector=json.loads(device_file.read_text())['gpu_uuid']if device_file.exists()else None
                    if selector is None:raise RuntimeError('actual child GPU UUID not yet available')
                    probe=subprocess.run(['nvidia-smi','-i',selector,'--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=5,check=True)
                    total=0;matches=0
                    for line in probe.stdout.splitlines():
                        pid,mib=line.split(',');pid=int(pid.strip())
                        if pid in host_tree:total+=int(mib.strip())*1048576;matches+=1
                    if matches:peak=total if peak is None else max(peak,total)
                    samples.append(dict(stage=name,monotonic=tick,tree_pids=sorted(tree),namespace_to_host_pid=mapping,gpu_selector=selector,raw_compute_apps=probe.stdout,matched_processes=matches,tree_used_bytes=total if matches else None))
                except Exception as e:failed+=1;samples.append(dict(stage=name,monotonic=tick,error=type(e).__name__))
                with (args.output/'memory_samples.private.jsonl').open('a')as f:f.write(json.dumps(samples[-1])+'\n')
                time.sleep(.5)
        record=dict(stage=name,exit_code=child.returncode,complete_child_wall_seconds=time.perf_counter()-start);records.append(record)
        (args.output/'process.private.json').write_text(json.dumps(records,indent=2)+'\n')
        if child.returncode!=0:raise RuntimeError('stage failed: '+name+'; all outputs retained')
    if args.gate_only:
        gaps=[b['monotonic']-a['monotonic']for a,b in zip(samples,samples[1:])if a['stage']==b['stage']]
        report=dict(status='complete_first_candidate_gate_only',records=records,observed_tree_gpu_peak_bytes=peak,failed_samples=failed,sample_count=len(samples),max_interval_seconds=max(gaps,default=None),requested_interval_seconds=.5,memory_budget_bytes=20_000_000_000,full_GA='not_assessed',whole_recon_all='not_assessed')
        (args.output/'monitor_gate.private.json').write_text(json.dumps(report,indent=2)+'\n')
        if peak is None:raise RuntimeError('actual host PID GPU memory attribution not observed')
        if peak>20_000_000_000:raise RuntimeError('observed GPU tree over 20 GB budget')
        (args.output/'GATE_MONITOR_COMPLETE.json').write_text(json.dumps(dict(status='complete',report_sha256=hashlib.sha256((args.output/'monitor_gate.private.json').read_bytes()).hexdigest()))+'\n')
        return
    comparisons={}
    for name in ['final_candidate.private.npz','final_state.private.npz','search.private.npz','all_scores.private.npz']:
        with np.load(args.output/'cpu'/name,allow_pickle=False)as a,np.load(args.output/'gpu'/name,allow_pickle=False)as b:
            fields={k:dict(shape_dtype_equal=a[k].shape==b[k].shape and a[k].dtype==b[k].dtype,bytes_equal=a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes())for k in set(a.files)&set(b.files)}
            comparisons[name]=dict(same_keys=set(a.files)==set(b.files),fields=fields,all_bytes_equal=set(a.files)==set(b.files)and all(v['bytes_equal']for v in fields.values()))
    for name in ['candidate_control.private.json','search_control.private.json']:comparisons[name]=dict(all_bytes_equal=json.loads((args.output/'cpu'/name).read_text())==json.loads((args.output/'gpu'/name).read_text()))
    gaps=[b['monotonic']-a['monotonic']for a,b in zip(samples,samples[1:])if a['stage']==b['stage']]
    passed=all(v['all_bytes_equal']for v in comparisons.values())
    report=dict(status='complete'if passed else'failed',records=records,comparisons=comparisons,all_numeric_state_and_control_equal=passed,tree_gpu_peak_bytes=peak,failed_memory_samples=failed,requested_memory_interval_seconds=.5,max_memory_interval_seconds=max(gaps,default=None),memory_sample_count=len(samples),memory_budget_bytes=20_000_000_000,memory_budget_pass=peak is not None and peak<=20_000_000_000,whole_recon_all='not_assessed',whole_GA='not_assessed',complete_first_defect=True,production_default_changed=False)
    path=args.output/'pair.private.json';path.write_text(json.dumps(report,indent=2)+'\n')
    if not passed:raise RuntimeError('complete first GA CPU/GPU differs; retained all diagnostics')
    (args.output/'COMPLETE.json').write_text(json.dumps(dict(status='complete',report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))+'\n')
if __name__=='__main__':main()
