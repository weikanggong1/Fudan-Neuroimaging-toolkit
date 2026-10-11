"""独立小型CUDA父子显存诊断；复用现有采样器，不推断可信host PID映射。"""
from __future__ import annotations
import argparse,contextlib,hashlib,json,os,pathlib,subprocess,sys,time,re


def identity():
    result={'local_pid':os.getpid(),'pid_namespace':None,'starttime_ticks':None,'nspid_chain':None,'errors':[]}
    try:
        result['pid_namespace']=os.readlink('/proc/self/ns/pid')
        result['starttime_ticks']=int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
        lines=pathlib.Path('/proc/self/status').read_text().splitlines()
        line=next((x for x in lines if x.startswith('NSpid:')),None)
        result['nspid_chain']=[int(x) for x in line.split()[1:]] if line else None
    except OSError as error:result['errors'].append(type(error).__name__)
    return result



def build_owned_sched_pid_map(query, gpu_uuid, local_pids, nvml_rows=None):
    """现场验证Linux3.10自有树的sched全局PID，返回(mapping,evidence)。

    query为已打开NvmlProcessQuery；gpu_uuid为完整目标UUID；local_pids只能
    属于调用者当前父子树；nvml_rows可为刚取得的同UUID快照，None自行查询。
    无法读身份/格式/名称/同卡驱动行时不映射，另记证据。无CUDA初始化、
    分配器或精度修改；不同内核须先独立验证来源，当前明确拒绝。
    """
    from fnit.recon_all.profiling import ProcessTreeDeviceSampler
    if not os.uname().release.startswith('3.10.'):
        raise ValueError('sched global PID source is validated here only for Linux 3.10')
    local_pids=tuple(local_pids)
    owned_tree=ProcessTreeDeviceSampler._tree(os.getpid())
    if not set(local_pids)<=owned_tree:raise ValueError('only currently owned process subtree allowed')
    raw=nvml_rows if nvml_rows is not None else query.snapshot(gpu_uuid)[0]
    evidence=[];mapping={}
    for local_pid in local_pids:
        initial=ProcessTreeDeviceSampler._process_identity(local_pid)
        path=pathlib.Path('/proc')/str(local_pid)
        try:
            first_line=next(iter((path/'sched').read_text().splitlines()),'')
            match=re.fullmatch(r'(.*) \((\d+), #threads: (\d+)\)',first_line)
            if not match:raise ValueError('unrecognized sched header')
            driver_pid=int(match[2]);comm=(path/'comm').read_text().strip()
            item={'local_pid':local_pid,'sched_driver_pid':driver_pid,'sched_comm_matches_local':match[1]==comm,'same_uuid_nvml_row_present':driver_pid in {row['pid'] for row in raw},'nvml_name_matches':False}
            if item['same_uuid_nvml_row_present']:
                try:
                    driver_name=query.module.nvmlSystemGetProcessName(driver_pid)
                    item['driver_pid_name_query_status']='available'
                except query.module.NVMLError as error:
                    item['driver_pid_name_query_status']=type(error).__name__
                # 名称API读取当前namespace的proc，使用自有local PID。
                name=query.module.nvmlSystemGetProcessName(local_pid)
                if isinstance(name,bytes):name=name.decode()
                name_parts=name.split()
                if not name_parts:raise ValueError('empty NVML local process name')
                basename=pathlib.Path(name_parts[0]).name
                item['nvml_process_name_basename']=basename;item['nvml_name_matches']=basename[:15]==comm
                item['nvml_name_query_scope']='local PID; name is not independent host ownership proof'
            final=ProcessTreeDeviceSampler._process_identity(local_pid)
            item['identity_unchanged']=initial==final and initial['identity_error'] is None
            if item['sched_comm_matches_local'] and item['same_uuid_nvml_row_present'] and item['nvml_name_matches'] and item['identity_unchanged']:
                mapping[driver_pid]={'local_pid':local_pid,'starttime_ticks':initial['starttime_ticks'],'pid_namespace':initial['pid_namespace'],'source':'owned proc sched header p->pid (Linux3.10 initial namespace), same UUID driver PID; local name and identity rechecked'}
                item['status']='verified_in_controlled_probe'
            else:item['status']='not_verified'
            evidence.append(item)
        except (OSError,ValueError,RuntimeError,query.module.NVMLError) as error:evidence.append({'local_pid':local_pid,'status':'unavailable','error_type':type(error).__name__})
    return mapping,evidence



def sample_current_call(sampler):
    """只使用本次新增sample；失败不会把历史ownership/合计带入当前帧。

    sampler为现有ProcessTreeDeviceSampler；forced调用前后分别记录样本和
    errors长度、monotonic跨度。返回(row或None, typed调用状态)，不导出错误
    文本或路径，不修改生产采样器。调用本身抛异常仍按原异常向外传播。
    """
    samples_before=len(sampler.samples);errors_before=len(sampler.errors)
    started=time.monotonic();sampler.sample_if_due(force=True);ended=time.monotonic()
    added=len(sampler.samples)-samples_before
    fresh_errors=sampler.errors[errors_before:]
    error_types=[]
    for error in fresh_errors:
        match=re.match(r'([A-Za-z_][\w.]*)\(',error.get('error',''))
        error_types.append(match[1] if match else 'unknown_error_type')
    row=sampler.samples[-1] if added>0 else None
    return row,{'new_sample_count':added,'new_error_count':len(fresh_errors),
      'new_error_types':error_types,'call_start_monotonic':started,
      'call_end_monotonic':ended,'call_span_seconds':ended-started,
      'status':'current_sample_available' if row is not None else 'unavailable',
      'sample_monotonic':row.get('monotonic') if row else None,
      'sample_query_seconds':row.get('sample_query_seconds') if row else None,
      'same_snapshot_attributed_processes':row.get('processes',[]) if row else [],
      'scope':'separate sampler query; never a direct-query snapshot or historical fallback'}


def correlate_snapshots(snapshots):
    """只派生分配阶段的时间相关候选；不能作为trusted_host_pid_map。"""
    frames={row['phase']:row for row in snapshots}
    pids=lambda phase:{row['pid'] for row in frames[phase]['nvml_processes']}
    return {'parent':sorted(pids('parent_allocate')-pids('before_cuda')),
            'child':sorted(pids('both_allocate')-pids('child_before_cuda')),
            'classification':'empirical temporal correlation only; not trusted host/local PID binding'}


def precision(torch):
    return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'autocast':torch.is_autocast_enabled()}


def worker(mib):
    import torch
    torch.set_num_threads(2)
    before=precision(torch);tensor=None
    def state():
        live=torch.cuda.is_initialized()
        return {'identity':identity(),'precision':precision(torch),'cuda_initialized':live,
          'allocator_disabled_environment_present':'PYTORCH_NO_CUDA_MEMORY_CACHING' in os.environ,'torch_stats_valid_for_peak':'PYTORCH_NO_CUDA_MEMORY_CACHING' not in os.environ,
          'torch_allocated_bytes':torch.cuda.memory_allocated() if live else None,
          'torch_reserved_bytes':torch.cuda.memory_reserved() if live else None,
          'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated() if live else None,
          'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved() if live else None,
          'payload_live_bytes':mib*2**20 if tensor is not None else 0,'monotonic':time.monotonic()}
    print(json.dumps(state()),flush=True)
    for command in sys.stdin:
        command=command.strip()
        if command=='init':torch.cuda.init();torch.cuda.set_device(0)
        elif command=='allocate':tensor=torch.empty(mib*2**20//4,dtype=torch.float32,device='cuda:0');tensor.zero_();torch.cuda.synchronize()
        elif command=='free':tensor=None;torch.cuda.synchronize()
        elif command=='flush':torch.cuda.empty_cache();torch.cuda.synchronize()
        elif command=='exit':assert precision(torch)==before;break
        else:raise ValueError('unknown child command')
        print(json.dumps(state()),flush=True)


def run(args):
    if args.output.exists():raise FileExistsError('output must not exist')
    args.output.mkdir(parents=True)
    if args.allocator=='enabled':os.environ.pop('PYTORCH_NO_CUDA_MEMORY_CACHING',None)
    elif args.allocator=='disabled':os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    import torch
    from fnit.recon_all.profiling import ProcessTreeDeviceSampler
    from fnit.recon_all.nvml_process_memory import NvmlProcessQuery
    torch.set_num_threads(2);before=precision(torch)
    query=NvmlProcessQuery();devices=query.devices();physical=next(k for k,v in devices.items() if v==args.gpu_uuid)
    rows,device_bytes,error=query.snapshot(args.gpu_uuid)
    handle=query.handles[args.gpu_uuid]
    graphics=query.module.nvmlDeviceGetGraphicsRunningProcesses(handle)
    utilization=query.module.nvmlDeviceGetUtilizationRates(handle)
    if rows or graphics or utilization.gpu!=0 or error or device_bytes!=args.idle_device_baseline_bytes:
        query.close();raise RuntimeError('target process/utilization or baseline idle gate failed; no CUDA allocation started')
    def permitted_proc():
        result=[]
        for path in ['/host/proc','/hostproc','/run/host/proc','/mnt/host/proc','/proc/1/root/proc']:
            try:status='directory' if pathlib.Path(path).is_dir() else 'absent'
            except OSError as error:status=type(error).__name__
            result.append({'candidate':path,'status':status})
        return result
    report={'scope':'small controlled allocation probe; not reconstruction/benchmark or 20GB bound',
      'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
      'monitor_source_sha256':{m:hashlib.sha256(pathlib.Path(sys.modules[m].__file__).read_bytes()).hexdigest() for m in ['fnit.recon_all.profiling','fnit.recon_all.nvml_process_memory']},
      'kernel_release':os.uname().release,'identity':identity(),'host_proc_candidates':permitted_proc(),
      'device':{'physical_index':int(physical),'uuid':args.gpu_uuid,'logical':'cuda:0'},
      'observed_idle_gate':{'compute_process_count':len(rows),'graphics_process_count':len(graphics),'gpu_utilization_percent':int(utilization.gpu),'device_used_bytes':device_bytes,'explicit_expected_baseline_bytes':args.idle_device_baseline_bytes},
      'sched_mapping_requested':args.sched_mapping,'allocator_requested':args.allocator,'precision_before':before,'parent_payload_mib':args.parent_mib,'child_payload_mib':args.child_mib,'snapshots':[]}
    try:report['pid1_namespace']={'status':'readable','value':os.readlink('/proc/1/ns/pid')}
    except OSError as error:report['pid1_namespace']={'status':type(error).__name__}
    report['proc_mounts']=[{'root':fields[3],'mount':fields[4]} for line in pathlib.Path('/proc/self/mountinfo').read_text().splitlines() if ' - proc ' in line for fields in [line.split()]]
    sampler=ProcessTreeDeviceSampler(device='cuda:0',parent_pid=os.getpid(),interval=.25,query_backend='nvml')
    tensor=None;child=None;child_state=None
    def parent_state():
        live=torch.cuda.is_initialized()
        return {'identity':identity(),'cuda_initialized':live,'torch_stats_valid_for_peak':'PYTORCH_NO_CUDA_MEMORY_CACHING' not in os.environ,'torch_allocated_bytes':torch.cuda.memory_allocated() if live else None,'torch_reserved_bytes':torch.cuda.memory_reserved() if live else None,'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated() if live else None,'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved() if live else None,'payload_live_bytes':args.parent_mib*2**20 if tensor is not None else 0}
    def snapshot(phase):
        time.sleep(.3);tick=time.monotonic();raw,used,err=query.snapshot(args.gpu_uuid);raw_ended=time.monotonic()
        mapping_started=raw_ended
        evidence=[];mapping={}
        if args.sched_mapping:
            mapping,evidence=build_owned_sched_pid_map(query,args.gpu_uuid,[os.getpid(),*([child.pid] if child is not None and child.poll() is None else [])],raw)
            sampler.trusted_host_pid_map=sampler._validate_pid_map(mapping)
        mapping_ended=time.monotonic()
        row,sampler_call=sample_current_call(sampler)
        state=parent_state();report['snapshots'].append({'phase':phase,'sched_mapping_evidence':evidence,'monotonic':tick,'raw_query_start_monotonic':tick,'raw_query_end_monotonic':raw_ended,'raw_query_span_seconds':raw_ended-tick,'raw_query_scope':'direct NVML query; not sampler query','mapping_span_seconds':mapping_ended-mapping_started,'sampler_call':sampler_call,'query_span_seconds':time.monotonic()-tick,'parent':state,'child':child_state,'nvml_processes':raw,'device_used_bytes':used,'device_error_type':err.split(':')[0] if err else None,
          'same_snapshot_nvml_all_process_sum_bytes':sum(x['bytes'] for x in raw),'attributed_tree_total_bytes':row['tree_total_bytes'] if row else None,'sampler_ownership':row['ownership'] if row else 'unavailable',
          'mapping_unavailable_reasons':row['pid_mapping']['unavailable_reasons'] if row else [],'sampler_mapping_status':row['pid_mapping']['status'] if row else None,
          'held_payload_total_bytes':state['payload_live_bytes']+(child_state['payload_live_bytes'] if child_state else 0)})
    def command(value):
        child.stdin.write(value+'\n');child.stdin.flush()
        line=child.stdout.readline()
        if not line:raise RuntimeError('child exited before response')
        return json.loads(line)
    try:
        snapshot('before_cuda')
        torch.cuda.init();torch.cuda.set_device(0)
        actual=str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')
        assert actual==args.gpu_uuid.lower().removeprefix('gpu-')
        snapshot('parent_context')
        tensor=torch.empty(args.parent_mib*2**20//4,dtype=torch.float32,device='cuda:0');tensor.zero_();torch.cuda.synchronize();snapshot('parent_allocate')
        with (args.output/'child.private.log').open('w') as log:
            child=subprocess.Popen([sys.executable,__file__,'--worker','--child-mib',str(args.child_mib)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
            child_state=json.loads(child.stdout.readline());snapshot('child_before_cuda')
            child_state=command('init');snapshot('child_context')
            child_state=command('allocate');snapshot('both_allocate')
            child_state=command('free');snapshot('child_free_cache_retained')
            child_state=command('flush');snapshot('child_flush_cache')
            child.stdin.write('exit\n');child.stdin.flush();assert child.wait(timeout=30)==0;child_state=None;snapshot('child_exited')
        tensor=None;torch.cuda.synchronize();snapshot('parent_free_cache_retained')
        torch.cuda.empty_cache();torch.cuda.synchronize();snapshot('parent_flush_cache')
        assert precision(torch)==before;report['precision_after']=precision(torch)
        report['status']='completed_sched_mapping_probe' if args.sched_mapping else 'completed_mapping_unresolved';report['trusted_host_pid_map_supplied']=args.sched_mapping
        # 唯一增量只描述相关性，不进入sampler的trusted映射。
        report['correlated_driver_candidates']=correlate_snapshots(report['snapshots'])
        report['sampler_summary']={'failed_sample_count':len(sampler.errors),'peak_tree_total_bytes':sampler.report()['peak_tree_total_bytes'],'status':sampler.report()['status']}
        report['ownership_conclusion']=('Explicit owned sched PID plus NVML UUID/name mapping evaluated at each snapshot; inspect per-row verification and tree sums; no global kernel or whole-memory guarantee' if args.sched_mapping else 'Host-side PID identity unavailable in initial permitted proc view; do not infer mapping from allocation deltas, numeric equality, or device-total memory')
        text=json.dumps(report,indent=2)+'\n';(args.output/'report.public.json').write_text(text)
        print(json.dumps({'status':report['status'],'snapshots':len(report['snapshots']),'correlated_driver_candidates':report['correlated_driver_candidates']}))
    finally:
        # 仅等待自己已启动的worker；不终止任何其他或共享服务器进程。
        if child is not None and child.poll() is None:
            with contextlib.suppress(BrokenPipeError):child.stdin.write('exit\n');child.stdin.flush()
            child.wait(timeout=30)
        sampler.close();query.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu-uuid');p.add_argument('--output',type=pathlib.Path);p.add_argument('--allocator',choices=['existing','enabled','disabled'],default='existing');p.add_argument('--parent-mib',type=int,default=128);p.add_argument('--child-mib',type=int,default=256);p.add_argument('--sched-mapping',action='store_true');p.add_argument('--idle-device-baseline-bytes',type=int,default=0);p.add_argument('--worker',action='store_true');args=p.parse_args()
    if not 1<=args.parent_mib<=256 or not 1<=args.child_mib<=256:p.error('payload per process must be 1–256 MiB')
    if args.worker:worker(args.child_mib)
    else:
        if not args.gpu_uuid or args.output is None:p.error('gpu-uuid and output required')
        run(args)

if __name__=='__main__':main()
