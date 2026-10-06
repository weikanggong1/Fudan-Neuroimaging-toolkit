"""Hash public scalar/proof/doc payloads; excludes arrays, binaries and own manifest."""
import hashlib
import json
from pathlib import Path

leaf=Path(__file__).resolve().parent
repo=leaf.parents[2]
files=sorted(p for p in leaf.rglob('*') if p.is_file() and p.name!='RESULTS_MANIFEST.json' and '__pycache__' not in p.parts)
files += [repo/'docs/synthseg/CPU_COLUMNS.md',repo/'src/fnit/synthseg_parc/CPU_COLUMNS_NOTICE.md']
rows={}
for path in files:
    if path.suffix in ('.nii','.gz','.npz','.npy','.h5','.so','.pyc'):raise RuntimeError('private science/binary payload prohibited')
    data=path.read_bytes();assert len(data)<2_000_000
    rows[str(path.relative_to(repo))]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
x={'schema':'fnit_columns_integration_final_public_manifest/v1','scope':'Source-bound scalar receipts, own validation code, docs and CC0 label PNG; no model/raw images/arrays/vendor binaries',
   'prepared_plan_sha256':'6daa84b1b18e6c343c493ef9bbff7e6fc83836c8b8149845caaef1f87655e8ed',
   'prepared_manifest_sha256':'8a01ce9da25c302bf16684d2c1945a5e8a8efe927979bdea4162e6ab961d9dcb',
   'count':len(rows),'files':rows}
(leaf/'RESULTS_MANIFEST.json').write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
print(json.dumps({'files':len(rows),'bytes':sum(v['bytes'] for v in rows.values()),'manifest_sha256':hashlib.sha256((leaf/'RESULTS_MANIFEST.json').read_bytes()).hexdigest()}))
