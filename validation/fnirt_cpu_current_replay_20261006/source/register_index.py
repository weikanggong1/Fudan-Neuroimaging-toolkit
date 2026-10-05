"""Register only this bounded independent leaf under all canonical index locks."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

ROOT=Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
KEY='fnirt_cpu_current_replay_20261006'
LEAF='smri_cpu_20261004/remaining_20261006/fnirt-current-pcg-replay-v1'
BASE='4f3267abac486314636d9b081a85b78833e62167'
LOCKS=['.INDEX.codex.lock','.index.lock','INDEX.json.lock','INDEX.md.lock',
       'admin/index-update.lock','admin/index.update.lock']

def atomic(path, data):
    before=path.stat()
    fd,temp=tempfile.mkstemp(prefix='.'+path.name+'.'+KEY+'.',dir=path.parent)
    try:
        os.fchmod(fd,stat.S_IMODE(before.st_mode))
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def main():
    handles=[]
    try:
        for name in sorted(LOCKS):
            path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
            fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
            handles.append(fd);fcntl.flock(fd,fcntl.LOCK_EX)
        # Always reread after all six locks; preserve every unrelated field.
        jp,mp=ROOT/'INDEX.json',ROOT/'INDEX.md'
        jbefore=jp.read_bytes();mbefore=mp.read_bytes()
        data=json.loads(jbefore);text=mbefore.decode()
        entry={'workspace':str(ROOT/'workspaces'/LEAF),'run_path':str(ROOT/'runs'/LEAF),
            'baseline_commit':BASE,'status':'bounded_arithmetic_research',
            'scope':'small arithmetic/guard contracts then one saved current H/g/diag system, old/new PCG pair with shared strict CSC; no assembly/MRI/reference process',
            'production_change':False,'gpu_enabled':False,'cpu_host':'nodecw7',
            'cpu_affinity':[32,36,40,44,48,52,56,60],'cpu_threads':8,
            'outer_lock':str(ROOT/'runs/smri_cpu_20261004/nodecw7.gems.cpu8.lock'),
            'runtime_constraint':'candidate uses own existing-dependency reductions; no installed FSL linking/calls',
            'child_timeout_seconds':180,'outer_lock_wait_seconds':1800}
        tasks=data.setdefault('active_tasks',{})
        if KEY in tasks and tasks[KEY]!=entry:raise RuntimeError('different existing own entry; no replacement')
        tasks[KEY]=entry
        data['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        heading='## '+KEY
        if heading not in text:
            text=text.rstrip()+'\n\n'+heading+'\n\n'+(
                '`workspaces/'+LEAF+'`, with outputs at the matching `runs/` leaf. '
                'Actual baseline `'+BASE+'`; small arithmetic/guard contracts before a single saved current H/g/diag old/new PCG pair, using shared strict CSC and unchanged 1e-3/500. '
                'Eight nodecw7 cores under the common CPU outer lock and a module inner lock. '
                'Independent candidate has no installed FSL link or runtime call; reference solution is only compared after solving. '
                'No production, GPU, MRI, assembly or full registration changes. Frozen inputs/results remain unchanged.\n')
        for group in ('workspaces','runs'):(ROOT/group/LEAF).mkdir(parents=True,exist_ok=True,mode=0o700)
        atomic(jp,(json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode())
        atomic(mp,text.encode())
        print(json.dumps({'entry':KEY,'baseline':BASE,'before_json_sha256':hashlib.sha256(jbefore).hexdigest(),
            'after_json_sha256':hashlib.sha256(jp.read_bytes()).hexdigest(),
            'before_md_sha256':hashlib.sha256(mbefore).hexdigest(),'after_md_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
            'six_locks':sorted(LOCKS),'permissions_preserved':True}))
    finally:
        for fd in reversed(handles):os.close(fd)

if __name__=='__main__':main()
