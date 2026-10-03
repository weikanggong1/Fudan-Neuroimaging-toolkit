from pathlib import Path
import sys, types, hashlib, importlib.util, json
import pytest
source = Path(sys.argv[1]).resolve()
for name, directory in [('fnit', source.parent), ('fnit.connectome', source)]:
    package=types.ModuleType(name); package.__path__=[str(directory)];sys.modules[name]=package
for name in ('fod','tracking'):
    spec=importlib.util.spec_from_file_location('fnit.connectome.'+name,source/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
print(json.dumps({'actual_source_path':str(source),'files_sha256':{name:hashlib.sha256((source/(name+'.py')).read_bytes()).hexdigest() for name in ('fod','tracking')}}),flush=True)
raise SystemExit(pytest.main(sys.argv[2:]))
