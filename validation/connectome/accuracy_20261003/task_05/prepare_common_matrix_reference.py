"""独立 CPU reference：真实固定 TCK 的逐轨节点及相同外部长度文件。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def checked(path,expected):
    path=Path(path)
    if sha(path)!=expected:raise ValueError(f'bytes changed: {path}')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bindings',required=True,type=Path)
    p.add_argument('--bindings-sha256',required=True)
    p.add_argument('--mrtrix-bin',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--case',default='sub-CON03')
    p.add_argument('--seed',default=0,type=int)
    args=p.parse_args()
    if args.output.exists():raise ValueError('fresh reference output required')
    args.output.mkdir(parents=True)
    bindings=json.loads(checked(args.bindings,args.bindings_sha256).read_text())
    entry=bindings['cases'][args.case]
    path=checked(entry['official_reference_manifest']['path'],entry['official_reference_manifest']['sha256'])
    producer=json.loads(path.read_text());root=path.parent
    if producer['state']!='completed' or producer['execution_completed'] is not True or len(producer['completed_commands'])!=198 or any(c['returncode']!=0 for c in producer['completed_commands']):
        raise ValueError('all original reference commands must actually complete')
    outputs=producer['outputs'][str(args.seed)];seed_root=root/f'seed-{args.seed}'
    inputs={}
    for name,expected in [('tracks.tck',outputs['tracks_sha256']),('sift2_weights.txt',outputs['scalars']['sift2_weights.txt']['sha256']),('lengths.txt',outputs['scalars']['lengths.txt']['sha256'])]:
        actual=checked(seed_root/name,expected);inputs[name]={'path':str(actual),'sha256':expected}
    binary=(args.mrtrix_bin/'tck2connectome').resolve()
    version=subprocess.check_output([str(binary),'-version'],text=True,stderr=subprocess.STDOUT)
    if '3.0.3-103-g026e850d' not in version:raise ValueError('exact official version required')
    program={'path':str(binary),'sha256':sha(binary),'version':version}
    result={'scope':'isolated official CPU fixed-TCK matrix/reference, not production or independent tracking',
        'case':args.case,'seed':args.seed,'host':socket.gethostname(),'harness_sha256':sha(__file__),
        'bindings_sha256':args.bindings_sha256,'producer':entry['official_reference_manifest'],
        'program':program,'inputs':inputs,'commands':[],'profiles':{},'state':'running','execution_completed':False,
        'length_definition':'same lengths.txt -scale_file in both arms; previous full chain -scale_length computes lengths directly and text export can round'}
    for profile,info in outputs['profiles'].items():
        atlas=checked(root/'inputs/atlases'/profile/'atlas_dwi.nii.gz',info['atlas_sha256'])
        destination=args.output/profile;destination.mkdir()
        profile_report={'atlas':{'path':str(atlas),'sha256':sha(atlas)},'outputs':{}}
        for kind in ['count','mean_length']:
            outfile=destination/(kind+'.csv')
            argv=[str(binary),'-symmetric','-assignment_radial_search','4']
            if kind=='count':argv+=['-out_assignments',str(destination/'assignments.txt')]
            else:argv+=['-tck_weights_in',inputs['sift2_weights.txt']['path'],'-scale_file',inputs['lengths.txt']['path'],'-stat_edge','mean']
            argv+=[inputs['tracks.tck']['path'],str(atlas),str(outfile),'-nthreads','8','-config','NIfTIUseSform','1']
            started=time.perf_counter()
            with open(destination/(kind+'.log'),'w') as log:
                process=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            elapsed=time.perf_counter()-started
            record={'profile':profile,'kind':kind,'argv':argv,'wall_s':elapsed,'returncode':process.returncode,
                'inputs':{**inputs,'atlas':profile_report['atlas']},'output':{'path':str(outfile),'sha256':sha(outfile) if outfile.exists() else None}}
            if kind=='count' and (destination/'assignments.txt').exists():record['assignments']={'path':str(destination/'assignments.txt'),'sha256':sha(destination/'assignments.txt')}
            result['commands'].append(record)
            if process.returncode:raise RuntimeError(f'official failed: {profile}/{kind}')
            profile_report['outputs'][kind]=record['output']
            if kind=='count':profile_report['outputs']['assignments']=record['assignments']
        checked(atlas,info['atlas_sha256'])
        result['profiles'][profile]=profile_report
        (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    for info in inputs.values():checked(info['path'],info['sha256'])
    checked(path,entry['official_reference_manifest']['sha256']);checked(args.bindings,args.bindings_sha256);checked(binary,program['sha256'])
    result.update(state='completed',execution_completed=True)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
