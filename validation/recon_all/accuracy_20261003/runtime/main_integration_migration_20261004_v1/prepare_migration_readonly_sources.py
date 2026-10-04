import subprocess,pathlib,json,hashlib,tarfile,io,ast,re,posixpath
ROOT='8eba2ace9b205d2611854ca71d80a85baf74659d';CAND='1be058a24b9ce702c644c13dddd7bd0f56d7c57f';PROD='3a0c9aba6321b4981fd8174b4b191515459aa38b'
SRC=pathlib.Path('/tmp/fnit-recon-accuracy-20261003/coordinator');DEST=pathlib.Path('/tmp/fnit-recon-accuracy-20261003/integration_20261004');REPORT=DEST/'validation/recon_all/accuracy_20261003/runtime/main_integration_migration_20261004_v1';REPORT.mkdir(parents=True,exist_ok=True)
git=lambda *a:subprocess.check_output(['git','-C',str(SRC),*a]);rf=set(git('ls-tree','-r','--name-only',ROOT).decode().splitlines());cf=set(git('ls-tree','-r','--name-only',CAND).decode().splitlines());assert subprocess.check_output(['git','-C',str(DEST),'rev-parse','HEAD'],text=True).strip()==CAND
# Only pinned tracked paths: entire accuracy evidence/tool addition set plus audited linked dependencies.
audit=json.loads((SRC/'validation/recon_all/accuracy_20261003/runtime/integration_dependency_audit_20261004_v1/audit.json').read_text())
selected={p for p in rf-cf if p.startswith('validation/recon_all/accuracy_20261003/')};selected|={x['path'] for x in audit['core_tools_transitive_closure'] if x['candidate_missing']};selected|={p for p in audit['document_referenced_tracked_paths_missing_candidate'] if p in rf and p.startswith(('validation/','docs/recon_all/'))}
selected|={x['path'] for x in audit['docs_recon_all_diff'] if x['status']=='A'}
assert all(not p.startswith(('src/','tools/','native/','install/')) for p in selected)
# Git archive exports exact blobs; root untracked working files never enter selected archive.
archive=git('archive',ROOT,*sorted(selected));records=[]
with tarfile.open(fileobj=io.BytesIO(archive)) as t:
 for m in t:
  if m.isfile():
   data=t.extractfile(m).read();target=DEST/m.name;assert not target.exists() or target.read_bytes()==data;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);records.append({'path':m.name,'action':'add_pinned_root','sha256':hashlib.sha256(data).hexdigest()})
read=lambda ref,p:git('show',ref+':'+p).decode();merged=[]
def save(p,text,reason):
 (DEST/p).write_text(text);merged.append({'path':p,'reason':reason,'candidate_sha256':hashlib.sha256(git('show',CAND+':'+p)).hexdigest(),'root_sha256':hashlib.sha256(git('show',ROOT+':'+p)).hexdigest(),'integrated_sha256':hashlib.sha256((DEST/p).read_bytes()).hexdigest()})
# These root diffs are evidence-only additions/corrections; individually inspected, preserve production definitions.
for name in ['CA_NORMALIZATION.md','SURFACE_STATS_CACHE.md','SYNTHSTRIP_MGH_DTYPE.md','NATIVE_PIAL_PLACEMENT.md']:
 p='docs/recon_all/'+name;save(p,read(ROOT,p),'Reviewed individual diff: add completed historical evidence/API table; no candidate production definition removed')
p='docs/recon_all/CUDA_STARTUP_RESOURCE_WAIT.md';s=read(ROOT,p);s=s.replace('本轮真实启动均一次成功，没有触发重试；原GPU0 OOM病例的空目录整例正在独立评估。','该两例annotation首次启动均成功，没有触发重试；后续8f及3a空目录整例已按独立回执评估，3a sub-06实际finish组双侧算法前OOM重启成功，见[候选整例报告](PRECISION_CANDIDATE_BENCHMARK_20261004.md)。不得将annotation的未重试范围推广到整例。');save(p,s,'Retain startup API and add genuine completed stage/whole-case recovery boundaries')
p='docs/recon_all/README.md';candidate=read(CAND,p);root=read(ROOT,p);intro=root.split('\n\n')[1:3];s=candidate.replace('# 单幅 T1w 的 recon-all 重建\n','# 单幅 T1w 的 recon-all 重建\n\n'+'\n\n'.join(intro)+'\n',1);s+='\n\n## 2026-10-04 本轮诊断与版本绑定\n\n[CUDA启动诊断](CUDA_BOOTSTRAP_DIAGNOSTICS.md)、[启动修复实测](CUDA_STARTUP_BENCHMARK_20261004.md)、[候选精度报告](PRECISION_CANDIDATE_BENCHMARK_20261004.md)保留各自工具/源码/实际运行SHA；集成工作分支尚未运行新算法，不将3a实测重标为集成版。候选原有完整CPU比较、半球缓存释放、分割统计、volmask与十例准备说明仍保留。\n';save(p,s,'Add measured precision status/navigation, preserve candidate CPU comparison/native availability/cache and segmentation docs')
for name in ['CONDA_CPP_BUILD.md','PROFILING.md','SPHERE_REGISTRATION_PERFORMANCE.md','THREAD_BUDGET.md']:
 p='docs/recon_all/'+name;save(p,read(CAND,p),'Reviewed root diff would regress newer candidate install16/parent-cache/CPU spherical averaging/failure-report semantics; retain candidate byte-for-byte')
p='validation/recon_all/accuracy_20261003/task_02/README.md';s=read(ROOT,p);s=s.replace('# 任务2：前段精度诊断与修复\n','# 任务2：前段精度诊断与修复\n\n本页所有前段结果对应历史816输入/该任务writer修复，未重标为3a或本集成分支的新整例；当前3a整例结果见主精度报告。\n',1);save(p,s,'Merge finished 10-pair prefix/N4 evidence, bind historical scope explicitly')
p='validation/recon_all/accuracy_20261003/task_03/README.md';quote=next(x for x in read(CAND,p).splitlines() if x.startswith('> 本页'));s=read(ROOT,p).replace('# GCA、归一化和 WM/filled 精度诊断\n','# GCA、归一化和 WM/filled 精度诊断\n\n'+quote.replace('新的十例连续链尚未运行','此句对应候选准备时点，后续3a真实连续链另见主报告')+'\n',1);save(p,s,'Preserve candidate historical version qualifier and add completed production-helper ABBA receipts')
p='validation/recon_all/accuracy_20261003/task_05/compare_placement.py';save(p,read(ROOT,p),'Reviewed read-only diagnostic augmentation: script hash, local CSV/cluster counts; no production or gate changes')
# Verify every candidate-only file retained byte-for-byte using Git blob hashing, including4docs.
preserved=sorted(cf-rf);bad=[]
tree={line.split('\t',1)[1]:line.split()[2] for line in git('ls-tree','-r',CAND).decode().splitlines()}
actuals=subprocess.check_output(['git','-C',str(DEST),'hash-object','--stdin-paths'],input=('\n'.join(preserved)+'\n').encode()).decode().splitlines()
for path,actual in zip(preserved,actuals):
 if tree[path]!=actual:bad.append(path)
assert not bad,bad
# src cannot be changed; candidate tree object equals measured3a, working paths unchanged.
srcdiff=subprocess.check_output(['git','-C',str(DEST),'diff','--name-only','--','src'],text=True);assert not srcdiff
assert git('rev-parse',CAND+':src')==git('rev-parse',PROD+':src')
for x in records:assert hashlib.sha256((DEST/x['path']).read_bytes()).hexdigest()==x['sha256']
parsed=[]
for item in audit['core_tools_transitive_closure']:
 p=DEST/item['path'];assert p.is_file();ast.parse(p.read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];parsed.append(item['path'])
# Check local relative links in actual integrated tracked/add docs, exclude external URLs/fragments.
links=[];missing=[]
for path in sorted({x['path'] for x in audit['docs_recon_all_diff'] if x['status']!='D' and x['path'].endswith('.md')}|{x['path'] for x in audit['accuracy_tool_docs']}):
 for target in re.findall(r'\]\(([^)]+)\)',(DEST/path).read_text()):
  if '://' in target or target.startswith('#'):continue
  clean=target.split('#')[0].strip('<>');local=(DEST/path).parent/clean
  if not local.exists():missing.append({'document':path,'target':target})
  links.append({'document':path,'target':target,'exists':local.exists()})
manifest={'status':'prepared_uncommitted_for_review','versions':{'root_source':ROOT,'candidate_base':CAND,'actual_measured_production':PROD},'integrated_branch':'recon-accuracy/20261004-main-integration','integrated_head_remains_candidate':CAND,'runtime_status':'No upload/compile/algorithm execution;3a observed runs remain3a, not this integration','added_pinned_tracked_files':records,'individually_reviewed_merges':merged,'candidate_only_preserved_count':len(preserved),'candidate_only_preserved_paths':preserved,'candidate_only_preserve_hash_check':'all Git blob hashes equal candidate','production_src_unchanged_and_equal3a':True,'source_tree_object':git('rev-parse',PROD+':src').decode().strip(),'core_tools_ast_sha_verified':parsed,'relative_links':links,'missing_relative_links':missing,'untracked_root_excluded':True,'tests':'pending focused CPU12+cancel1'}
(REPORT/'migration_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'added':len(records),'reviewed_merges':len(merged),'candidate_preserved':len(preserved),'tools_verified':len(parsed),'links':len(links),'missing_links':missing},ensure_ascii=False))
