"""只读CPU比较失败816检查点与已完成启动候选的现存共同前缀；不是整例等效。"""
from __future__ import annotations
import argparse
from collections import Counter
from numbers import Integral
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import tarfile
import time
import traceback

VOLUMES = ('orig/001.mgz','rawavg.mgz','orig.mgz','nu.mgz','T1.mgz','brainmask.mgz',
           'norm.mgz','brain.mgz','wm.seg.mgz','wm.asegedit.mgz','wm.mgz','filled.mgz','aseg.presurf.mgz')
DISCRETE = {'wm.seg.mgz','wm.asegedit.mgz','wm.mgz','filled.mgz','aseg.presurf.mgz'}
SURFACES = ('orig.nofix','orig.premesh','orig','white.preaparc','smoothwm','smoothwm.nofix',
            'inflated','inflated.nofix','qsphere.nofix','sphere','sphere.reg')
MAPS = ('curv','sulc','avg_curv','white.H','white.K','inflated.H','inflated.K')


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def file_binding(reference,candidate):
    result={'reference_path':str(reference),'candidate_path':str(candidate),
            'reference_exists':reference.is_file(),'candidate_exists':candidate.is_file()}
    for role,path in (('reference',reference),('candidate',candidate)):
        if path.is_file():result[role+'_sha256']=digest(path)
    result['common']=result['reference_exists'] and result['candidate_exists']
    result['shared_file_sha_equal']=(result['reference_sha256']==result['candidate_sha256']) if result['common'] else None
    return result


def bound_json(path, expected_sha=None):
    data=Path(path).read_bytes()
    sha=hashlib.sha256(data).hexdigest()
    if expected_sha is not None and sha!=expected_sha:
        raise ValueError('frozen metadata SHA differs: '+str(path))
    value=json.loads(data)
    if not isinstance(value,dict):raise ValueError('metadata must be an object: '+str(path))
    return value,sha


def bind_run(config_path,archive_path,expected_commit,role,*,expected_completion_sha256,
             expected_pipeline_sha256,baseline_failure_stage='annotation_hemisphere_group'):
    config_path,archive_path=Path(config_path),Path(archive_path)
    config,config_sha=bound_json(config_path);diagnostics=Path(config['diagnostic_root'])
    launch_path,completion_path=diagnostics/'launch.json',diagnostics/'completion.json'
    pipeline_path=Path(config['output'])/'fnit-native-free-run.json'
    for value in (expected_completion_sha256,expected_pipeline_sha256):
        if not isinstance(value,str) or len(value)!=64 or any(char not in '0123456789abcdef' for char in value):
            raise ValueError('frozen completion/pipeline SHA256 must be declared')
    launch,launch_sha=bound_json(launch_path)
    completion,completion_sha=bound_json(completion_path,expected_completion_sha256)
    pipeline,pipeline_sha=bound_json(pipeline_path,expected_pipeline_sha256)
    if config['code_commit']!=expected_commit or launch['config_sha256']!=config_sha:
        raise ValueError(role+' actual config/commit/launch binding differs')
    for key,value in config.items():
        if launch.get(key)!=value:raise ValueError(role+' config/launch field differs: '+key)
    if not launch.get('started_utc') or launch.get('status')=='prepared_not_executed':
        raise ValueError(role+' is a preparation, not an actual execution')
    if launch['host']!=socket.gethostname() or type(config['threads']) is not int or config['threads']!=4 or config['device']!='cuda:0':
        raise ValueError(role+' host/threads/device differs')
    environment=launch.get('environment',{})
    if (environment.get('CUDA_VISIBLE_DEVICES')!=config['gpu_uuid'] or environment.get('OMP_NUM_THREADS')!='4'
            or environment.get('PYTORCH_NO_CUDA_MEMORY_CACHING')!='1'
            or Path(environment.get('PYTHONPATH','')).resolve()!=(Path(config['code_root'])/'src').resolve()):
        raise ValueError(role+' actual GPU/thread/cache/source launch environment differs')
    if completion['code_commit']!=config['code_commit'] or completion['source_archive_sha256']!=config['source_archive_sha256']:
        raise ValueError(role+' completion source binding differs')
    # Current real cases are CLI. API cannot be certified using a guessed argv shape.
    if config.get('invocation')!='cli':raise ValueError(role+' common-prefix binding currently requires actual CLI protocol')
    command=launch.get('command')
    expected_command=[config['python'],'-m','fnit.recon_all.native_free',config['input'],config['output'],
        '--weights-dir',config['weights'],'--assets-dir',config['assets'],'--native-bin-dir',config['native_bin_dir'],
        '--device',config['device'],'--threads','4','--profile-stages','--cuda-allocator-cache','disabled',
        *config.get('pipeline_cli_args',[])]
    if command!=expected_command:raise ValueError(role+' launch command input/output/resource protocol differs')
    if (pipeline.get('input')!=config['input'] or pipeline.get('subject_dir')!=config['output']
            or type(pipeline.get('threads')) is not int or pipeline['threads']!=4 or pipeline.get('device')!=config['device']):
        raise ValueError(role+' pipeline input/subject_dir/thread/device differs')
    precision=pipeline.get('precision',{})
    for key,expected in (('matmul_tf32_default',True),('cudnn_tf32_default',True),
                         ('fp16_or_bf16_requested_by_fnit',False),('fp16_or_bf16_enabled',False)):
        if precision.get(key) is not expected:raise ValueError(role+' pipeline precision differs: '+key)
    for device in ('cpu','cuda'):
        if precision.get('caller_autocast',{}).get(device,{}).get('enabled') is not False:
            raise ValueError(role+' pipeline caller autocast differs: '+device)
    if pipeline.get('gpu_memory_mode')!='disabled' or pipeline.get('cuda_allocator',{}).get('effective')!='disabled':
        raise ValueError(role+' pipeline actual allocator policy differs')
    exits=[completion.get(key) for key in ('exit_code','child_exit_code')]
    if any(type(value) is not int for value in exits):
        raise ValueError(role+' completion exit_code/child_exit_code must be strict integers')
    total=pipeline.get('total_seconds');command_seconds=completion.get('command_seconds')
    if any(type(value) not in (int,float) or not math.isfinite(value) or value<=0 for value in (total,command_seconds)) or command_seconds<total:
        raise ValueError(role+' completion/pipeline duration state differs')
    failure_evidence=None
    if role=='baseline':
        if completion.get('execution_status')!='failed' or any(value==0 for value in exits) or exits[0]!=exits[1] or pipeline.get('status')!='failed':
            raise ValueError('baseline must bind the original failed execution, not a completed whole case')
        if (pipeline.get('failed_stage')!=baseline_failure_stage or not pipeline.get('error')
                or not completion.get('error')):
            raise ValueError('baseline actual failure stage/error differs')
        stages=pipeline.get('stages',[])
        failure_rows=[row for row in stages if row.get('name')==baseline_failure_stage and row.get('error')]
        groups=[row for row in pipeline.get('hemisphere_scheduling',{}).get('groups',[])
                if row.get('operation')=='annotation' and row.get('status')=='failed']
        if (not failure_rows or failure_rows[-1]['error']!=pipeline['error'] or not groups
                or not any(row.get('status')=='failed' and row.get('error') and row.get('traceback')
                           for row in groups[-1].get('workers',{}).values())):
            raise ValueError('baseline pipeline annotation failure lacks matching stage/worker evidence')
        if 'pipeline_status' in completion and completion['pipeline_status']!=pipeline['status']:
            raise ValueError('baseline completion/pipeline status differs')
        failure_evidence={'failed_stage':pipeline['failed_stage'],'error':pipeline['error'],
                          'stage':failure_rows[-1],'worker_group':groups[-1]}
    else:
        if (completion.get('execution_status')!='complete' or any(value!=0 for value in exits)
                or completion.get('pipeline_status')!='complete' or pipeline.get('status')!='complete'):
            raise ValueError('candidate must finish before common-prefix comparison')
        if pipeline.get('failed_stage') or pipeline.get('error') or any(row.get('error') for row in pipeline.get('stages',[])):
            raise ValueError('candidate complete pipeline contains failure evidence')
        validation=pipeline.get('output_validation',{})
        if (validation.get('status')!='passed' or type(validation.get('expected')) is not int
                or validation['expected']!=138 or type(validation.get('present')) is not int
                or validation['present']!=138 or validation.get('missing')!=[]):
            raise ValueError('candidate pipeline output_validation is not passed 138/138')
        if completion.get('output_validation')!=validation or completion.get('pipeline_total_seconds')!=total:
            raise ValueError('candidate completion/pipeline validation or total_seconds differs')
        mesh=pipeline.get('mesh_validation',{})
        if mesh.get('status')!='passed' or any(mesh.get(hemi,{}).get('status')!='passed' for hemi in ('lh','rh')):
            raise ValueError('candidate pipeline mesh validation is not passed')
        numeric=pipeline.get('numeric_validation')
        if not isinstance(numeric,dict) or numeric.get('status') not in ('not_run','passed'):
            raise ValueError('candidate pipeline numeric validation state is missing or failed')
        if (completion.get('strict_reproduction')!='not_assessed'
                or completion.get('new_degradation')!='not_assessed'
                or not isinstance(completion.get('overall_metric_equivalence'),str)
                or not completion['overall_metric_equivalence'].startswith('not_assessed')):
            raise ValueError('candidate completion precision/equivalence state exceeds recorded scope')
        if 'numeric_validation' in completion and completion['numeric_validation']!=numeric:
            raise ValueError('candidate completion/pipeline numeric validation differs')
        outputs=pipeline.get('outputs')
        if not isinstance(outputs,dict) or len(outputs)!=138:raise ValueError('candidate pipeline output path manifest is incomplete')
        for relative,value in outputs.items():
            name=Path(relative)
            if name.is_absolute() or '..' in name.parts or str(Path(config['output'])/name)!=value:
                raise ValueError('candidate pipeline output path belongs to another run: '+relative)
            if not Path(value).is_file():raise FileNotFoundError('candidate declared output missing: '+value)
    if digest(config['input'])!=config['input_sha256']:
        raise ValueError(role+' raw input SHA differs')
    if digest(archive_path)!=config['source_archive_sha256']:
        raise ValueError(role+' source archive SHA differs')
    source=Path(config['code_root']);archived={}
    with tarfile.open(archive_path,'r:*') as archive:
        for member in archive:
            name=Path(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise ValueError('unsafe source archive member: '+member.name)
            if not member.isfile():continue
            result=hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
            archived[str(name)]=result.hexdigest()
            if digest(source/name)!=archived[str(name)]:raise ValueError(role+' archive/source differs: '+str(source/name))
    critical=('src/fnit/recon_all/native_free.py','src/fnit/recon_all/hemisphere_worker.py','src/fnit/recon_all/hemisphere_parallel.py')
    if any(name not in archived for name in critical):raise ValueError(role+' archive misses core source')
    if launch['candidate_native_free_sha256']!=archived[critical[0]]:raise ValueError(role+' native_free SHA differs')
    actual_python={str(path.relative_to(source)) for path in source.rglob('*.py') if '.git' not in path.parts and '__pycache__' not in path.parts}
    if actual_python!={name for name in archived if name.endswith('.py')}:raise ValueError(role+' source Python set differs')
    return config,{'role':role,'execution_status':completion['execution_status'],
        'whole_case_complete':role!='baseline','completion':completion,
        'config_path':str(config_path),'launch_path':str(launch_path),'completion_path':str(completion_path),
        'pipeline_path':str(pipeline_path),'config_sha256':config_sha,'launch_sha256':launch_sha,
        'completion_sha256':completion_sha,'pipeline_sha256':pipeline_sha,
        'pipeline_status':pipeline['status'],'precision':precision,'pipeline_total_seconds':total,
        'output_validation':pipeline.get('output_validation'),'mesh_validation':pipeline.get('mesh_validation'),
        'numeric_validation':pipeline.get('numeric_validation'),'failure_evidence':failure_evidence,
        'code_commit':config['code_commit'],'source_archive_sha256':digest(archive_path),
        'critical_source_sha256':{name:archived[name] for name in critical},'subject':config['output'],
        'input_sha256':config['input_sha256'],'host':launch['host'],'gpu_uuid':config['gpu_uuid'],'threads':4}


def tool(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def array_difference(reference,candidate,np):
    row={'reference_shape':list(reference.shape),'candidate_shape':list(candidate.shape),
         'reference_dtype':str(reference.dtype),'candidate_dtype':str(candidate.dtype)}
    if reference.shape!=candidate.shape:
        row['status']='not_assessed_shape_mismatch';return row
    finite=np.isfinite(reference)&np.isfinite(candidate)
    row['nonfinite_elements']=int((~finite).sum())
    row['different_elements']=int(np.count_nonzero(reference!=candidate))
    difference=np.abs(reference.astype(np.float64)-candidate.astype(np.float64))[finite]
    row['max_absolute_difference']=float(difference.max()) if difference.size else None
    row['p99_absolute_difference']=float(np.percentile(difference,99)) if difference.size else None
    return row


def cortex_membership(reference_ids,candidate_ids,vertex_count):
    """Official fix-ga may append repeated IDs; compare counts and membership.

    Do not truncate floats or deduplicate the files. Counts retain every entry.
    """
    if type(vertex_count) is not int or vertex_count<=0:
        raise ValueError('cortex anchor vertex count must be a positive integer')
    result={};counters=[]
    for role,ids in (('reference',reference_ids),('candidate',candidate_ids)):
        values=list(ids)
        if any(isinstance(value,bool) or not isinstance(value,Integral) or value<0 or value>=vertex_count for value in values):
            raise ValueError('cortex label IDs must be integers inside the proven anchor range: '+role)
        counts=Counter(int(value) for value in values);counters.append(counts)
        result[role]={'entries':len(values),'unique_vertices':len(counts),
                      'duplicate_entries':len(values)-len(counts),
                      'repeated_vertex_count':sum(count>1 for count in counts.values())}
        result[role+'_counts']=[counts.get(vertex,0) for vertex in range(vertex_count)]
    a,z=(set(counter) for counter in counters);total=len(a)+len(z)
    result['label_vertex_dice']=float(2*len(a&z)/total) if total else 1.
    result['different_vertex_memberships']=len(a^z)
    return result


def read_cortex_ids(path,fs,np):
    # Nibabel's integer read must not silently turn a fractional raw ID into an ID.
    lines=Path(path).read_text().splitlines()
    if len(lines)<2:raise ValueError('cortex label header is missing')
    declared=int(lines[1].strip());records=[line for line in lines[2:] if line.strip()]
    ids=[]
    for line in records:
        try:ids.append(int(line.split()[0]))
        except (ValueError,IndexError) as error:raise ValueError('cortex raw vertex ID is not an integer') from error
    if declared<0 or declared!=len(ids):raise ValueError('cortex label declared entry count differs')
    loaded=np.asarray(fs.read_label(str(path)))
    if loaded.ndim!=1 or not np.issubdtype(loaded.dtype,np.integer) or not np.array_equal(loaded,np.asarray(ids)):
        raise ValueError('cortex reader/raw integer ID sequence differs')
    return loaded


def surface_gate(reference,candidate,anchor_pairs,frame,topology,fs,np):
    pairs=[fs.read_geometry(str(path),read_metadata=True) for path in (reference,candidate)]
    vertices=[pair[0] for pair in pairs];faces=[pair[1] for pair in pairs];metadata=[pair[2] for pair in pairs]
    valid_arrays=all(x.ndim==2 and x.shape[1]==3 and len(x)>0 and np.isfinite(x).all()
        and f.ndim==2 and f.shape[1]==3 and len(f)>0 and np.issubdtype(f.dtype,np.integer)
        and np.all((f>=0)&(f<len(x))) for x,f in zip(vertices,faces))
    topologies=[topology(x,f) for x,f in zip(vertices,faces)] if valid_arrays else [{'valid':False}]*2
    for row in topologies:row['valid']=bool(row['valid'])
    ordered=valid_arrays and vertices[0].shape==vertices[1].shape and np.array_equal(faces[0],faces[1])
    anchor=bool(anchor_pairs) and valid_arrays and all(x.shape==a.shape and np.array_equal(f,af)
        for (x,f),(a,af) in zip(zip(vertices,faces),anchor_pairs))
    footer=all(np.array_equal(metadata[0].get(key),metadata[1].get(key))
               for key in set(metadata[0])|set(metadata[1]) if key!='filename')
    gate=bool(frame and ordered and anchor and footer and all(row['valid'] for row in topologies))
    row={'reference_shape':list(vertices[0].shape),'candidate_shape':list(vertices[1].shape),
         'reference_dtype':str(vertices[0].dtype),'candidate_dtype':str(vertices[1].dtype),
         'reference_faces_shape':list(faces[0].shape),'candidate_faces_shape':list(faces[1].shape),
         'reference_faces_dtype':str(faces[0].dtype),'candidate_faces_dtype':str(faces[1].dtype),
         'ordered_faces_equal':bool(ordered),'matches_own_white_preaparc_anchor':bool(anchor),
         'surface_footer_equal':bool(footer),'surface_frame_equal':bool(frame),'topology':topologies,
         'vertex_correspondence':gate,'indexed_difference':None}
    if gate:
        row['indexed_difference']=array_difference(vertices[0],vertices[1],np)
        distance=np.linalg.norm(vertices[0]-vertices[1],axis=1)
        row['indexed_difference'].update(different_vertices=int(np.count_nonzero(distance)),
            max_vertex_distance_mm=float(distance.max()),p99_vertex_distance_mm=float(np.percentile(distance,99)))
    else:row['status']='not_assessed_indexed_vertices_without_correspondence'
    return row,vertices


def compare(config,out,report):
    import nibabel as nib
    import nibabel.freesurfer.io as fs
    import numpy as np
    os.environ['PYTHONPATH']=str(Path(config['comparison_code_root'])/'src')
    sys.path.insert(0,os.environ['PYTHONPATH'])
    import fnit.recon_all.compare_subject as topology_module
    if Path(topology_module.__file__).resolve()!=(Path(config['comparison_code_root'])/'src/fnit/recon_all/compare_subject.py').resolve():
        raise ValueError('actual imported topology comparator differs from frozen comparison_code_root')
    _topology=topology_module._topology
    directory=Path(config['scripts_dir']);volume=tool(directory/'compare_volume_prefix.py','existing_volume_prefix')
    report['comparator_sha256']={str(path):digest(path) for path in (
        directory/'compare_volume_prefix.py',Path(config['comparison_code_root'])/'src/fnit/recon_all/compare_subject.py')}
    baseline,bb=bind_run(config['baseline_config'],config['baseline_archive'],config['baseline_commit'],'baseline',
        expected_completion_sha256=config['baseline_completion_sha256'],expected_pipeline_sha256=config['baseline_pipeline_sha256'])
    candidate,cb=bind_run(config['candidate_config'],config['candidate_archive'],config['candidate_commit'],'candidate',
        expected_completion_sha256=config['candidate_completion_sha256'],expected_pipeline_sha256=config['candidate_pipeline_sha256'])
    if baseline['input']!=candidate['input'] or bb['input_sha256']!=cb['input_sha256'] or bb['gpu_uuid']!=cb['gpu_uuid'] or bb['precision']!=cb['precision']:
        raise ValueError('raw input, GPU or precision differs between actual executions')
    report['execution_binding']={'baseline_failed_checkpoint':bb,'startup_only_candidate':cb}
    reference,candidate_root=Path(baseline['output']),Path(candidate['output'])
    for root in (reference,candidate_root):
        if not root.is_dir():raise FileNotFoundError('actual subject directory unavailable: '+str(root))
    report['files']={};rows=report['files']
    def record(relative,operation):
        left,right=reference/relative,candidate_root/relative
        row=file_binding(left,right);rows[relative]=row
        initial_reference_sha=row.get('reference_sha256');initial_candidate_sha=row.get('candidate_sha256')
        if row['common']:
            try:row.update(operation(left,right))
            except Exception as error:row.update(status='file_comparison_failed',error=repr(error),traceback=traceback.format_exc())
            if digest(left)!=initial_reference_sha or digest(right)!=initial_candidate_sha:
                raise ValueError('artifact changed during CPU comparison: '+relative)
        write(out/'common_prefix.json',report)
    for name in VOLUMES:
        def image(left,right,name=name):
            a,z=nib.load(str(left)),nib.load(str(right));aa,zz=np.asarray(a.dataobj),np.asarray(z.dataobj)
            if not all(np.isfinite(x).all() for x in (aa,zz,a.affine,z.affine)):
                raise ValueError('nonfinite volume or affine; existing comparator not applied')
            row={'reference_affine':a.affine.tolist(),'candidate_affine':z.affine.tolist(),
                 'reference_shape':list(aa.shape),'candidate_shape':list(zz.shape),
                 'reference_dtype':str(aa.dtype),'candidate_dtype':str(zz.dtype),
                 'same_voxel_grid':bool(aa.shape==zz.shape and np.array_equal(a.affine,z.affine)),
                 'affine_difference':array_difference(a.affine,z.affine,np)}
            if not row['same_voxel_grid']:
                row['status']='not_assessed_voxel_difference_without_same_grid';return row
            labels=name in DISCRETE
            if labels and not all(np.isfinite(x).all() and np.all(x==np.floor(x)) and np.all(x>=0) for x in (aa,zz)):
                raise ValueError('declared discrete volume contains noninteger/negative/nonfinite values')
            row.update(volume.compare_volume(left,right,labels));return row
        record('mri/'+name,image)
    for name in ('talairach.lta','talairach_with_skull.lta'):
        def lta(left,right):
            row=volume.compare_lta(left,right)
            if not all(np.isfinite(np.asarray(row[key])).all() for key in ('reference_matrix','candidate_matrix')):
                raise ValueError('nonfinite LTA matrix; numerical summary refused')
            difference=np.abs(np.asarray(row['reference_matrix'])-np.asarray(row['candidate_matrix']))
            row.update(reference_shape=[4,4],candidate_shape=[4,4],reference_dtype='float64',candidate_dtype='float64',
                       p99_absolute_element_difference=float(np.percentile(difference,99)),
                       matrix_scope='raw LTA 4x4 entries; transform header semantics are not asserted');return row
        record('mri/transforms/'+name,lta)
    orig_paths=[root/'mri/orig.mgz' for root in (reference,candidate_root)]
    frame=False
    if all(path.is_file() for path in orig_paths):
        a,z=[nib.load(str(path)) for path in orig_paths]
        frame=bool(a.shape==z.shape and np.array_equal(a.affine,z.affine)
                   and np.array_equal(a.header.get_vox2ras_tkr(),z.header.get_vox2ras_tkr()))
    report['surface_frame']={'common_conform_affine_and_tkras':frame,'affine_tolerance':0}
    for hemi in ('lh','rh'):
        anchors=[root/'surf'/f'{hemi}.white.preaparc' for root in (reference,candidate_root)]
        anchor_pairs=[fs.read_geometry(str(path)) for path in anchors] if all(path.is_file() for path in anchors) else None
        for stage in SURFACES:
            record('surf/'+hemi+'.'+stage,lambda left,right:surface_gate(left,right,anchor_pairs,frame,_topology,fs,np)[0])
        extra_maps=sorted({path.name[len(hemi)+1:] for root in (reference,candidate_root)
                           for path in (root/'surf').glob(hemi+'.smoothwm.*.crv')})
        for name in (*MAPS,*extra_maps):
            anchor_name='inflated' if name in ('sulc','inflated.H','inflated.K') else 'white.preaparc'
            gate=rows.get('surf/'+hemi+'.'+anchor_name,{}).get('vertex_correspondence') is True
            def morph(left,right,gate=gate):
                a,z=[fs.read_morph_data(str(path)) for path in (left,right)]
                row={'reference_shape':list(a.shape),'candidate_shape':list(z.shape),'reference_dtype':str(a.dtype),
                     'candidate_dtype':str(z.dtype),'vertex_correspondence':gate,
                     'stage_scope':'current shared artifact; may have been overwritten by candidate finishing stages'}
                if not gate:row['status']='not_assessed_map_without_vertex_correspondence';return row
                if not anchor_pairs or any(x.ndim!=1 or len(x)!=len(anchor[0]) for x,anchor in zip((a,z),anchor_pairs)):
                    row['status']='not_assessed_map_vertex_count_mismatch';return row
                row.update(array_difference(a,z,np));return row
            record('surf/'+hemi+'.'+name,morph)
        def cortex(left,right,hemi=hemi,anchor_pairs=anchor_pairs):
            a,z=[read_cortex_ids(path,fs,np) for path in (left,right)]
            gate=rows.get('surf/'+hemi+'.white.preaparc',{}).get('vertex_correspondence') is True
            if any(np.any(ids<0) for ids in (a,z)):
                raise ValueError('cortex label has negative vertex IDs')
            if anchor_pairs and any(np.any(ids>=len(anchor[0])) for ids,anchor in zip((a,z),anchor_pairs)):
                raise ValueError('cortex label vertex IDs exceed their own anchor range')
            row={'reference_shape':list(a.shape),'candidate_shape':list(z.shape),'reference_dtype':str(a.dtype),
                 'candidate_dtype':str(z.dtype),'vertex_correspondence':gate,
                 'index_sequence_equal':bool(np.array_equal(a,z)),
                 'label_semantics':'fix-ga concatenation; per-vertex multiplicity and idempotent membership'}
            if gate:
                if not anchor_pairs or len(anchor_pairs[0][0])!=len(anchor_pairs[1][0]):
                    raise ValueError('cortex label lacks a common proven anchor')
                membership=cortex_membership(a,z,len(anchor_pairs[0][0]))
                reference_counts=np.asarray(membership.pop('reference_counts'),dtype=np.int64)
                candidate_counts=np.asarray(membership.pop('candidate_counts'),dtype=np.int64)
                row.update(membership)
                row['multiplicity_difference']=array_difference(reference_counts,candidate_counts,np)
                row['membership_difference']=array_difference(reference_counts>0,candidate_counts>0,np)
            else:row['status']='not_assessed_label_without_vertex_correspondence'
            return row
        record('label/'+hemi+'.cortex.label',cortex)
    for relative,row in rows.items():
        for role,root in (('reference',reference),('candidate',candidate_root)):
            if row.get(role+'_exists') and digest(root/relative)!=row[role+'_sha256']:
                raise ValueError('artifact changed before final receipt: '+relative)
    for binding in (bb,cb):
        for name in ('config','launch','completion','pipeline'):
            if digest(binding[name+'_path'])!=binding[name+'_sha256']:
                raise ValueError('execution receipt changed: '+binding[name+'_path'])
    report['summary']={'compared_common_files':sum(row['common'] for row in rows.values()),
        'missing_on_either_side':sum(not row['common'] for row in rows.values()),
        'numeric_not_assessed_files':sum(row.get('status','').startswith('not_assessed') for row in rows.values()),
        'file_comparison_errors':sum(row.get('status')=='file_comparison_failed' for row in rows.values()),
        'scope':'current artifacts only; failed baseline remains incomplete; no whole-case or precision equivalence'}
    report['status']='comparison_complete' if not report['summary']['file_comparison_errors'] else 'comparison_errors'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args(argv);config=read(args.config);out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    report={'status':'starting','scope':'failed baseline common prefix vs startup-only candidate; CPU only',
            'config_sha256':digest(args.config),'script_sha256':digest(__file__),'evaluated_role':'startup_only_candidate',
            'baseline_whole_case_complete':False,'overall_metric_equivalence':'not_assessed'}
    start=time.monotonic();path=out/'common_prefix.json';write(path,report)
    try:
        index=Path(config.get('fnit_index','/cwStorage/home/gongwk/Notebook_code/FNIT/INDEX.json'))
        read(index);report['fnit_index_sha256']=digest(index)
        with Path(config['lock']).open('a+') as lock:
            wait=time.monotonic();fcntl.flock(lock,fcntl.LOCK_EX)
            report['lock_wait_seconds']=time.monotonic()-wait
            compare(config,out,report)
    except BaseException as error:
        report.update(status='comparison_failed',error=repr(error)+'; '+str(error),traceback=traceback.format_exc())
    finally:
        report['total_wall_seconds']=time.monotonic()-start;write(path,report)
    return 0 if report['status']=='comparison_complete' else 1


if __name__=='__main__':raise SystemExit(main())
