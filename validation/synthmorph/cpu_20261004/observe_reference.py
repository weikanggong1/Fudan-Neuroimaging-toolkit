"""Time original reference Python boundaries without changing calculations.

Run only with FreeSurfer's independent reference interpreter. Profiled clocks
are diagnostic, and must not be substituted for uninstrumented CLI timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);p.add_argument('--report',required=True);p.add_argument('arguments',nargs=argparse.REMAINDER)
    args=p.parse_args();arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    active={};rows=[];source_files=set()
    def callback(frame,event,arg):
        if event=='call':
            name=frame.f_code.co_name;filename=frame.f_code.co_filename;cls=type(frame.f_locals.get('self')).__name__
            category=None
            if filename.endswith('/synthmorph/registration.py') and name in ('register','network_space','transform','load_weights'):
                category='registration.'+name
            elif '/voxelmorph/tf/networks.py' in filename and name=='__init__' and cls in ('VxmAffineFeatureDetector','HyperVxmJoint'):
                category='network_construct.'+cls
            elif '/keras/' in filename and name=='__call__' and cls in ('VxmAffineFeatureDetector','HyperVxmJoint'):
                category='network_call.'+cls
            elif '/surfa/' in filename and name in ('load_volume','save_volume','transform'):
                category='surfa.'+name
            if category:
                active[id(frame)]=(category,time.perf_counter());source_files.add(filename)
        elif event=='return' and id(frame) in active:
            category,start=active.pop(id(frame));rows.append({'boundary':category,'seconds':time.perf_counter()-start})
    sys.argv=[args.entry]+arguments;sys.setprofile(callback);start=time.perf_counter();status=0
    try:runpy.run_path(args.entry,run_name='__main__')
    except SystemExit as error:
        status=0 if error.code is None else error.code
        if status:raise
    finally:
        elapsed=time.perf_counter()-start;sys.setprofile(None)
        tf=sys.modules.get('tensorflow');sf=sys.modules.get('surfa');vxm=sys.modules.get('voxelmorph')
        data={'scope':'original FreeSurfer Python observer; inclusive boundaries, nested clocks cannot be summed; not formal speed timing','observer_wall_seconds':elapsed,'status':status,'entry_sha256':hashlib.sha256(Path(args.entry).read_bytes()).hexdigest(),'boundaries':rows,'versions':{name:getattr(module,'__version__',None) for name,module in [('tensorflow',tf),('surfa',sf),('voxelmorph',vxm)]},'source_sha256':{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_files if Path(path).is_file()}}
        Path(args.report).write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
