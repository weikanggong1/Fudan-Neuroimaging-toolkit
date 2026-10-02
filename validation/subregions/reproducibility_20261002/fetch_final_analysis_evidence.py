"""Fetch saved final metrics and metadata only, with exact size/SHA verification."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import shlex,subprocess,sys,tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('first-stage','first-raw','complete'),required=True)
    parser.add_argument('--remote-helper',type=Path,default=Path('/tmp/fnit_subregions_remote.py'))
    parser.add_argument('--remote-root',default='/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent)
    args=parser.parse_args()
    archive_path=Path('/tmp')/('fnit_final_'+args.phase+'_derived_20261002.tar.gz')
    manifest_name='final_'+args.phase.replace('-','_')+'_fetch_manifest.json'
    remote=r'''import hashlib,io,json,sys,tarfile
from pathlib import Path
root=Path(REMOTE_ROOT);d=root/'reproducibility_20261002';analysis=d/'final_all_analysis'
phase=PHASE
queue=json.loads((d/'final_all_release_queue.json').read_text())
if phase=='complete':
 manager=json.loads((analysis/'cpu_manager.json').read_text())
 if manager['state']!='completed' or queue['state']!='completed':raise ValueError('Final analysis and six GPU runs must be complete')
 result=json.loads((analysis/'final_full_analysis_status.json').read_text())
 if result['state']!='completed' or result['groups']!=24:raise ValueError('Final result must contain 24 groups')
 timing=json.loads((analysis/'timing_complete/v2_status.json').read_text())
 if timing['state']!='completed':raise ValueError('Complete nested recipe timer extraction must be complete')
 paths=[(path,'final_all_analysis/'+str(path.relative_to(analysis))) for path in sorted(analysis.rglob('*')) if path.is_file() and path.suffix in ('.json','.tsv','.log')]
 runs=[run for run in queue['runs'] if run.get('phase')=='verified_full_run']
 if len(runs)!=6:raise ValueError('Six completed full runs required')
else:
 mode=phase.removeprefix('first-')
 status=json.loads((analysis/f'first_{mode}_r1_status.json').read_text())
 if status['state']!='completed':raise ValueError('First-run CPU result not completed')
 paths=[(analysis/name,'final_all_analysis/'+name) for name in (f'first_{mode}_r1_status.json',f'first_{mode}_r1_cross.json',f'first_{mode}.log','cpu_manager.json')]
 runs=[run for run in queue['runs'] if run.get('phase')=='verified_full_run' and run['mode']==mode and run['repeat']==1]
 if len(runs)!=1:raise ValueError('One verified first full run required')
paths += [(d/name,name) for name in ('final_all_release_queue.json','final_expected_source_manifest.json','official_explicit_step_timers.json','reconall_historical_lineage_audit.json')]
if phase=='complete':
 artifact=d/'final_all_release_artifacts.json'
 if artifact.exists():paths.append((artifact,artifact.name))
for run in runs:
 output=Path(run['output'])
 for name in ('report.json','api_report.json','context_identity.json','comparison.tsv'):
  paths.append((output/name,'final_all_runs/'+output.name+'/'+name))
 for name in (output.name+'_gpu_load.jsonl',output.name+'.log'):
  paths.append((d/name,'final_all_runs/'+output.name+'/'+('gpu_load.jsonl' if name.endswith('.jsonl') else 'driver.log')))
# Cache exact bytes before hashing and archiving, including mutable queue snapshots.
blobs=[(name,path.read_bytes()) for path,name in paths]
if len({name for name,_ in blobs})!=len(blobs):raise ValueError('Duplicate derived file paths')
records=[{'path':name,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()} for name,blob in blobs]
manifest={'scope':'Derived saved metrics/JSON/TSV/log/GPU/context metadata only; no image/model/atlas/license content','phase':phase,'files':records}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name,blob in blobs:
  info=tarfile.TarInfo(name);info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
 blob=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo(MANIFEST_NAME);info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
'''.replace('REMOTE_ROOT',repr(args.remote_root)).replace('PHASE',repr(args.phase)).replace('MANIFEST_NAME',repr(manifest_name))
    with archive_path.open('wb') as handle:
        process=subprocess.run([sys.executable,str(args.remote_helper),shlex.join(['python','-c',remote])],stdout=handle,stderr=subprocess.PIPE)
    if process.returncode:raise RuntimeError(process.stderr.decode('utf-8',errors='replace')[-3000:])
    destination=args.output_dir.resolve();destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive_path,'r:gz') as tar:
        members=tar.getmembers()
        if len({member.name for member in members})!=len(members):raise ValueError('Duplicate archive member names')
        for member in members:
            if not member.isfile() or not (destination/member.name).resolve().is_relative_to(destination):raise ValueError('Unsafe member: '+member.name)
        manifest=json.load(tar.extractfile(manifest_name));records={r['path']:r for r in manifest['files']}
        if set(records)!={m.name for m in members if m.name!=manifest_name}:raise ValueError('Manifest/member set differs')
        for member in members:
            raw=tar.extractfile(member).read()
            if member.name!=manifest_name:
                record=records[member.name]
                if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Transfer size/SHA differs: '+member.name)
            path=destination/member.name;path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix(path.suffix+'.download.tmp');temporary.write_bytes(raw);temporary.replace(path)
    output={'phase':args.phase,'archive_bytes':archive_path.stat().st_size,'archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'verified_records':len(manifest['files']),'manifest_sha256':hashlib.sha256((destination/manifest_name).read_bytes()).hexdigest()}
    print(json.dumps(output))


if __name__=='__main__':main()
