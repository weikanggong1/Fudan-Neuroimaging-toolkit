import ast,array,collections,csv,hashlib,json,math,pathlib,struct,sys,zipfile
R=pathlib.Path('/tmp/fnit-startup-stage-audit-v2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readnpz(p):
 out={}
 with zipfile.ZipFile(p) as z:
  for n in z.namelist():
   b=z.read(n);assert b[:6]==b'\x93NUMPY'
   version=b[6:8];hlen=struct.unpack('<H' if version==b'\x01\x00' else '<I',b[8:10] if version==b'\x01\x00' else b[8:12])[0];offset=10 if version==b'\x01\x00' else 12
   h=ast.literal_eval(b[offset:offset+hlen].decode('latin1').strip());assert not h['fortran_order'];raw=b[offset+hlen:]
   out[n[:-4]]={'header':h,'raw':raw,'sha256':hashlib.sha256(raw).hexdigest()}
 return out
def ints(v):
 assert v['header']['descr']=='<i8';a=array.array('q');a.frombytes(v['raw'])
 if sys.byteorder!='little':a.byteswap()
 assert len(a)==math.prod(v['header']['shape']);return a
A={'archive_sha256':'eb2784227f79bb6914147f85702c1d0669d29ed1cfd4510c7e3adfbf8d4f02e4','scope':'CPU archive review; NPZ directly revalidated, original annot/mesh absent from archive; no GPU/no rerun','cases':{},'gaps':[]}
for case in ('ds000114_sub-06','ds000114_sub-07'):
 pairpath=R/(case+'_comparison')/'annotation.pair.json';p=json.load(open(pairpath));caseout={'pair_sha256':sha(pairpath),'exact_regression_pass':p['exact_regression_pass'],'all_gates':p['gates'],'meshes_reported':p['meshes'],'annotations':{},'arms':{}}
 stages={}
 for arm in ('baseline','candidate'):
  q=R/(case+'_'+arm);s=json.load(open(q/'outputs/annotation.stage.json'));g=json.load(open(q/'outputs/annotation.group.json'));m=json.load(open(q/'diagnostics/monitor/monitor.json'));c=json.load(open(q/'diagnostics/controller.json'));stages[arm]=s
  rows=list(csv.DictReader(open(q/'diagnostics/monitor/gpu_samples.csv')));own=max(int(x['total_process_bytes']) for x in rows if x['apps_status']=='ok');card=max(int(x['gpu_total_mib'])*1048576 for x in rows if x['gpu_status']=='ok')
  assert own==m['peak_sampled_process_bytes'];assert len(rows)==m['samples'];assert c['all_tracked_owned_exited'];assert s['status']=='stage_complete' and c['status']=='complete' and s['exit_code']==m['exit_code']==c['exit_code']==0
  caseout['arms'][arm]={'stage_status':s['status'],'controller_status':c['status'],'exit_code':0,'all_owned_exited':c['all_tracked_owned_exited'],'gpu_uuid_declared':s['gpu_uuid_declared'],'gpu_uuid_worker_actual':s['worker_device_validation'],'gpu_parent_actual':s.get('parent_actual_device'),'initialized_parent':s['initialized_parent'],'precision':s['actual_parent_precision'],'worker_precision':{h:w['precision'] for h,w in g['workers'].items()},'threads':s['threads'],'runtime':s['runtime'],'source':c['source'],'source_files_changed_during_stage':s['changed_imported_sources'],
   'timings_seconds':{'stage_validation_through_summary':s['timings']['validation_through_summary_write_seconds'],'group_call_wall':s['timings']['group_wall_seconds'],'group_internal_wall':g['group_wall_seconds'],'monitor_command_wall':m['command_wall_seconds'],'controller_monitored_wall':c['monitored_process_wall_seconds'],'outer_validation_admission_cleanup_wall':c['outer_wall_seconds_through_cleanup_and_validation'],'admission_wait':c['admission_wait_seconds'],'group_startup_wait_actual':g.get('startup_wait_seconds_actual'),'group_worker_span':g['worker_span_seconds'],'operation_worker_span':g.get('operation_worker_span_seconds'),'worker_operation':{h:w['stage']['seconds'] for h,w in g['workers'].items()},'stage_breakdown':s['timings']},
   'sampling_bytes':{'whole_stage_own_tree_peak':own,'group_only_own_tree_peak':g['device_process_tree']['peak_tree_total_bytes'],'whole_card_peak':card,'scope':'same-query parent+descendants total, separate whole-card query; both sampled, not continuous or allocated tensor peak','continuous_peak_verified':m['continuous_peak_verified'],'samples':m['samples'],'requested_interval_seconds':m['sampling_interval_requested_seconds'],'max_gap_seconds':m['maximum_sampling_gap_seconds'],'group_max_gap_seconds':g['device_process_tree']['max_observed_interval_seconds'],'failed_app_queries':m['failed_app_queries'],'csv_failed_gpu_queries':sum(x['gpu_status']!='ok' for x in rows),'torch_memory_stats_status':s['torch_memory_stats_status']},'file_sha':{'stage':sha(q/'outputs/annotation.stage.json'),'group':sha(q/'outputs/annotation.group.json'),'monitor':sha(q/'diagnostics/monitor/monitor.json'),'controller':sha(q/'diagnostics/controller.json')}}
 for key, row in p['annotations'].items():
  vectors=[]
  for arm in ('baseline','candidate'):
   s=stages[arm];f=R/(case+'_'+arm)/'outputs/annotation_semantics'/(key+'.npz');rec=s['annotations'][key]['semantic_vectors'];assert sha(f)==rec['sha256'] and f.stat().st_size==rec['size_bytes'];v=readnpz(f)
   assert set(v)=={'original_annotation_ids','label_table_indices','vertex_indices','color_table','names'}
   for field, attr in [('original_annotation_ids','label_original_ids_sha256_le_i64'),('label_table_indices','label_table_indices_sha256_le_i64'),('color_table','color_table_sha256_le_i64')]:assert v[field]['sha256']==s['annotations'][key][attr]
   vertices=ints(v['vertex_indices']);assert list(vertices)==list(range(len(vertices)))
   vectors.append(v)
  x,y=vectors;assert all(x[n]['header']==y[n]['header'] and x[n]['raw']==y[n]['raw'] for n in x)
  ids,ids2=ints(x['original_annotation_ids']),ints(y['original_annotation_ids']);left,right=collections.Counter(ids),collections.Counter(ids2);both=collections.Counter(a for a,b in zip(ids,ids2) if a==b)
  dice={i:2*both[i]/(left[i]+right[i]) for i in left.keys()|right.keys()}
  assert all(v==1 for v in dice.values());assert all(dice[v['packed_original_id']]==v['dice'] and left[v['packed_original_id']]==v['baseline_vertices'] for v in row['dice_original_ids']['labels'])
  assert len(ids)==stages['baseline']['annotations'][key]['vertex_count']
  caseout['annotations'][key]={'vertices':len(ids),'labels_present':len(dice),'all_per_label_dice':row['dice_original_ids']['labels'],'dice_min':min(dice.values()),'dice_p05':row['dice_original_ids']['p05'],'dice_median':row['dice_original_ids']['median'],'different_original_id_vertices':sum(a!=b for a,b in zip(ids,ids2)),'different_table_index_vertices':row['different_table_index_vertices'],'npz_sha_independently_verified':True,'all_npz_arrays_exact_independently_verified':True,'annotation_bytes_same_reported':row['annotation_file_bytes_same'],'npz_bytes_same':row['npz_file_bytes_same'],'arrays_exact_reported':row['arrays_exact']}
 A['cases'][case]=caseout
A['gaps']=['Archive has no completion.json; controller complete + stage_complete + monitor exit0 provide stage completion, not whole-case completion.','Original .annot and smoothwm/sphere.reg files are not included: bytes/geometry claims audited from bound reports, cannot independently reread original mesh here.','No tensor allocator continuous peak (cache disabled); sampled NVML peaks are lower bounds and differ across stage/group sampler cadences.','Single annotation pair per case on GPU1; timing changes cannot establish speedup or historical GPU0 OOM mitigation.','No official accuracy or original-T1 whole-case chain in this stage scope.']
(R/'cpu_annotation_stage_audit.json').write_text(json.dumps(A,indent=2)+'\n')
print(json.dumps({c:{a:{'timings':v['timings_seconds'],'memory':v['sampling_bytes']} for a,v in x['arms'].items()} for c,x in A['cases'].items()},indent=2))
