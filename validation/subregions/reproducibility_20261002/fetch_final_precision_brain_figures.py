"""Fetch only final derived PNG/JSON with explicit size/SHA verification."""
from __future__ import annotations
import argparse,base64,hashlib,json,shlex,struct,subprocess,zlib
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-folder',required=True)
    parser.add_argument('--local-folder',type=Path,required=True)
    parser.add_argument('--prefix',default='final_brain_figures')
    args=parser.parse_args()
    if Path(args.prefix).name!=args.prefix or args.prefix in ('.','..'):
        raise ValueError('prefix must be one filename component')
    code=r'''from pathlib import Path
import base64,hashlib,json,zlib
folder=Path(REMOTE_FOLDER);prefix=PREFIX
files=[]
for mode in ('stage','raw'):
 manifest_path=folder/prefix/mode/'figures_manifest.json'
 if not manifest_path.exists(): continue
 manifest=json.loads(manifest_path.read_text())
 for entry in manifest['files']:
  relative=Path(entry['path']);path=folder/relative
  if relative.parts[:2]!=(prefix,mode) or relative.suffix not in ('.png','.json') or '..' in relative.parts:
   raise ValueError('Unapproved derived figure path')
  data=path.read_bytes()
  if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
   raise ValueError('Completed figure bytes changed')
  files.append((entry['path'],data))
 files.append((str(manifest_path.relative_to(folder)),manifest_path.read_bytes()))
 revision=folder/prefix/mode/'raw_display_window_revision.json'
 if revision.exists(): files.append((str(revision.relative_to(folder)),revision.read_bytes()))
for name in ('final_figures_raw_launch.json','final_figures_stage_launch.json','final_figures_renderer_upload.json','final_raw_window_launch.json'):
 path=folder/name
 if path.exists():files.append((name,path.read_bytes()))
print(json.dumps({'files':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'content':base64.b64encode(zlib.compress(data)).decode()} for name,data in files]}))
'''.replace('REMOTE_FOLDER',repr(args.remote_folder)).replace('PREFIX',repr(args.prefix))
    remote='/home1/gongwk/anaconda3/bin/python -c '+shlex.quote(code)
    response=subprocess.run(['python3','/tmp/fnit_subregions_remote.py',remote],check=True,capture_output=True,text=True)
    payload=json.loads(response.stdout)
    result={'scope':'derived final figures and metadata only; no imaging volumes, posterior arrays, weights, atlas or licenses transferred','files':[]}
    for entry in payload['files']:
        relative=Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe path')
        data=zlib.decompress(base64.b64decode(entry.pop('content')))
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Transfer bytes/SHA mismatch')
        if relative.suffix=='.png':
            if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Invalid PNG')
            entry['figure_dimensions_pixels']=list(struct.unpack('>II',data[16:24]))
        path=args.local_folder/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        if path.stat().st_size!=entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Saved file identity mismatch')
        result['files'].append(entry)
    result['PNG_count']=sum(Path(e['path']).suffix=='.png' for e in result['files'])
    target=args.local_folder/'final_figures_fetch_manifest.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified_files':len(result['files']),'PNG_count':result['PNG_count'],'manifest':str(target)}))


if __name__=='__main__':main()
