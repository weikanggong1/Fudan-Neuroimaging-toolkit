"""Inside fixed official image: read completed node results; zero imaging APIs."""
import argparse
import ast
import fcntl
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re

from volume_numeric import sha

STAGES=(('register_template_wf','anatomical_standard_normalization'),
        ('brain_extraction_wf','anatomical_brain_extraction'),('brain_seg_wf','anatomical_segmentation'),
        ('bold_hmc_wf','motion'),('bold_reg_wf','bbr'),('bold_anat_wf','T1w_preproc'),
        ('bold_std_wf','MNI_preproc'),('bold_MNI6_wf','MNI_preproc'),
        ('bold_native_wf','native_preproc'),('bold_boldref_wf','reference'),
        ('hmc_boldref_wf','reference'),('bold_confounds_wf','confounds'))


def read(path): return json.loads(Path(path).read_text())


def fields(value):
    if isinstance(value,dict): return {str(k):fields(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [fields(v) for v in value]
    if value is None or isinstance(value,(str,int,bool)): return value
    if isinstance(value,float): return value if math.isfinite(value) else str(value)
    return str(value)


def files(value):
    if isinstance(value,dict):
        for x in value.values(): yield from files(x)
    elif isinstance(value,(list,tuple)):
        for x in value: yield from files(x)
    elif isinstance(value,str) and Path(value).is_file(): yield value


def save(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def interval_union(bounds):
    merged=[]
    for start,end in sorted(bounds):
        if end < start: raise ValueError('Node runtime end precedes start')
        if merged and start <= merged[-1][1]: merged[-1][1]=max(end,merged[-1][1])
        else: merged.append([start,end])
    return sum((b-a).total_seconds() for a,b in merged)


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    p.add_argument('--job-index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True)
    p.add_argument('--wrapper-source',type=Path,required=True);p.add_argument('--wrapper-sha256',required=True)
    p.add_argument('--cpu-binding',type=Path,required=True);p.add_argument('--cpu-binding-sha256',required=True)
    args=p.parse_args();os.umask(0o077)
    if args.output.exists(): raise FileExistsError('Preserve previous node export')
    if sha(args.cpu_binding)!=args.cpu_binding_sha256:raise ValueError('Final CPU resource binding differs')
    cpu=read(args.cpu_binding);pair=cpu['cpu_pairs']['fnirt_cpu1']
    if pair['threads']!=1 or len(pair['affinity'])!=1:raise ValueError('Reuse an existing authorized CPU1 core')
    os.sched_setaffinity(0,set(pair['affinity']))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    leases=[]
    for key,mode in (('queue_lock',fcntl.LOCK_EX),('shared_group_lease',fcntl.LOCK_SH)):
        path=Path(pair[key])
        if not path.is_file():raise FileNotFoundError('Reuse existing CPU1 leases')
        handle=path.open('a+');fcntl.flock(handle,mode);leases.append(handle)
    if (sha(args.manifest)!=args.manifest_sha256 or sha(args.inputs)!=args.inputs_sha256
            or sha(args.wrapper_source)!=args.wrapper_sha256): raise ValueError('Pinned official evidence differs')
    if importlib.metadata.version('fmriprep')!='25.2.4': raise ValueError('Use declared official25.2.4 image')
    config=read(args.inputs);manifest=read(args.manifest);job=manifest['jobs'][args.job_index]
    if job['function']!='volume' or job['backend']!='official': raise ValueError('Complete official volume job required')
    image_sha=sha(config['fmriprep_image'])
    if image_sha!=config['expected_fmriprep_sha256']: raise ValueError('Actual official image SHA differs')
    root=Path(job['output_dir']); report_path=root/'report.safe.json';report=read(report_path)
    if (report['status']!='complete' or report['source_unchanged'] is not True
            or report['adapter_sha256']!=manifest['adapter_sha256'] or report['cpu_threads']!=job['threads']
            or report['cpu_affinity']!=list(map(int,job['cpu_list'].split(','))) or len(report['records'])!=2):
        raise ValueError('Wait for both original first/cache processes to complete')
    command=job['command'];source=Path(command[command.index('--source')+1])
    actual_source={x.relative_to(source).as_posix():sha(x) for x in (source/'src/fnit').rglob('*.py')}
    if actual_source!=report['source_sha256']: raise ValueError('Official adapter source snapshot changed')
    wrapper_ast=ast.parse(args.wrapper_source.read_text())
    wrapper=next(ast.literal_eval(n.value) for n in wrapper_ast.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='_WRAPPER' for t in n.targets))
    wrapper_sha=hashlib.sha256(wrapper.encode()).hexdigest()
    receipts=[]
    for repeat,row in enumerate(report['records']):
        path=root/f'repeat_{repeat}'/'native_child_exit.private.json';receipt=read(path)
        argv=receipt['original_argv']; entry=Path(argv[0])
        if (row['repeat']!=repeat or row['native_exit_code']!=0 or row['native_payload_exit_code']!=0
                or row['native_exit_accepted'] is not True or receipt['payload_exit_code']!=0
                or sha(entry)!=receipt['payload_entrypoint_sha256']
                or row['native_entrypoint_sha256']!=receipt['payload_entrypoint_sha256']
                or row['reference_wrapper_sha256']!=wrapper_sha): raise ValueError('Native launcher/payload/source proof differs')
        if ('--fs-no-reconall' not in argv or '--no-msm' not in argv or '--ignore' not in argv
                or 'slicetiming' not in argv[argv.index('--ignore')+1:]
                or 'fieldmaps' not in argv[argv.index('--ignore')+1:]
                or argv[argv.index('--nthreads')+1]!=str(job['threads'])
                or argv[argv.index('--omp-nthreads')+1]!=str(job['threads'])
                or argv[argv.index('--dummy-scans')+1]!='0'
                or 'T1w' not in argv or 'MNI152NLin6Asym:res-2' not in argv):
            raise ValueError('Actual original volume/STC-OFF/full-frame CPU protocol differs')
        raw=config['public180']
        for label,key in (('raw_BOLD','bold'),('raw_T1w','t1w')):
            path_input=Path(raw[key]); expected=report['input_sha256'][label]
            if sha(path_input)!=expected['sha256'] or path_input.stat().st_size!=expected['bytes']:
                raise ValueError('Original complete raw inputs changed')
        log=root/f'repeat_{repeat}'/'native.private.log'
        cached_lines=re.findall(r'^.*\[Node\].*(?:Cached|cached).*$|^.*[Cc]ached.*\[Node\].*$',log.read_text(errors='replace'),re.M)
        receipts.append({'repeat':repeat,'call_type':row['call_type'],'complete_frames':row['complete_frames'],
                         'official_wall_seconds_including_IO':row['official_wall_seconds_including_io'],
                         'payload_wall_seconds':receipt['payload_wall_seconds'],
                         'receipt_sha256':sha(path),'native_log_sha256':sha(log),'cached_node_log_messages':len(cached_lines),
                         'native_launcher_exit':0,'native_payload_exit':0,'native_entrypoint_sha256':sha(entry),
                         'original_argv_sha256':hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest()})
    from nipype.utils.filemanip import loadpkl
    nodes=[]; inventory=[]; seen=set(); all_bounds=[]
    for path in sorted((root/'work').rglob('result_*.pklz')):
        result=loadpkl(str(path));runtime=result.runtime;parent=isinstance(runtime,(list,tuple))
        operations=runtime if parent else [runtime]
        stage=next((label for needle,label in STAGES if needle in str(path)),'other_original_nodes')
        outputs=result.outputs.trait_get() if hasattr(result.outputs,'trait_get') else vars(result.outputs)
        outputs=fields(outputs)
        inventory.append({'result_path':str(path),'result_sha256':sha(path),'stage':stage,
                          'operation':path.name,'output_fields':outputs,'observed_files':list(files(outputs))})
        if parent: continue
        for r in operations:
            duration=getattr(r,'duration',None)
            if duration is not None and (not math.isfinite(float(duration)) or duration<0): raise ValueError('Invalid completed node duration')
            start=getattr(r,'startTime',None);end=getattr(r,'endTime',None);cmd=getattr(r,'cmdline',None)
            key=(start,end,duration,cmd)
            if key in seen: continue
            seen.add(key)
            bounds=(datetime.fromisoformat(start),datetime.fromisoformat(end)) if start and end else None
            if bounds: all_bounds.append(bounds)
            nodes.append({'stage':stage,'operation_name_sha256':hashlib.sha256(path.name.encode()).hexdigest(),
                          'duration_seconds':None if duration is None else float(duration),
                          '_bounds':bounds,'command_sha256':hashlib.sha256((cmd or '').encode()).hexdigest()})
    if not nodes: raise ValueError('No actual completed original leaf nodes')
    origin=min(a for a,b in all_bounds) if all_bounds else None
    stage_summary={}
    for stage in sorted({x['stage'] for x in nodes}):
        rows=[x for x in nodes if x['stage']==stage];bounds=[x['_bounds'] for x in rows if x['_bounds']]
        stage_summary[stage]={'leaf_executions':len(rows),
                              'leaf_duration_sum_seconds':sum(x['duration_seconds'] for x in rows if x['duration_seconds'] is not None),
                              'observed_active_interval_union_seconds':interval_union(bounds),
                              'elapsed_span_seconds':(max(b for a,b in bounds)-min(a for a,b in bounds)).total_seconds() if bounds else None}
    for n in nodes:
        bounds=n.pop('_bounds');n['interval_seconds_after_first_node']=None if bounds is None else [
            (x-origin).total_seconds() for x in bounds]
    derivative_files=[str(x) for x in (root/'derivatives').rglob('*') if x.is_file()]
    public={'schema_version':1,'status':'completed_original_saved_nodes_exported','new_imaging_API_calls':0,
            'fmriprep_version':'25.2.4','actual_image_sha256':image_sha,'reference_wrapper_source_sha256':sha(args.wrapper_source),
            'wrapper_payload_code_sha256':wrapper_sha,'adapter_sha256':report['adapter_sha256'],
            'manifest_sha256':args.manifest_sha256,'inputs_configuration_sha256':args.inputs_sha256,
            'execution_report_sha256':sha(report_path),'source_tree_python_unchanged':True,
            'raw_input_hashes':{key:report['input_sha256'][key] for key in ('raw_BOLD','raw_T1w')},
            'cpu_threads':job['threads'],'calls':receipts,'stage_groups':stage_summary,'leaf_nodes':nodes,
            'whole_scope':'Complete original raw-BIDS volume preproc,including confounds and required anatomical template registrations; no PICA/AROMA/FNITclean/recon-all/surface.',
            'cache_scope':'Both original invocations share one work root. Second process may reuse first-call workflow nodes; saved node runtimes can retain first-call intervals. Original first/cache process clocks remain distinct.',
            'node_time_scope':'Deduplicated completed leaf runtimes. MapNode parents excluded; stage duration sums and unions overlap and do not replace whole process time.',
            'GNU_time_scope':'Not present in the declared v4 command. No global GNU user/system/RSS inferred from Nipype nodes.'}
    args.output.mkdir(parents=True);save(args.output/'nodes.public.json',public)
    save(args.output/'inventory.private.json',{'job':job,'node_outputs':inventory,'derivative_files':derivative_files})
    print(json.dumps({'status':public['status'],'nodes_report_sha256':sha(args.output/'nodes.public.json'),
                      'inventory_sha256':sha(args.output/'inventory.private.json'),'new_imaging_API_calls':0}))


if __name__=='__main__':main()
