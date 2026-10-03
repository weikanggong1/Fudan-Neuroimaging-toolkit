#!/usr/bin/env python3
"""Headcw CPU read-only file/geometry/schema audit. Never calls a solver."""
import hashlib, io, json, socket, sys, time, unittest
from pathlib import Path
from datetime import datetime, timezone
import nibabel as nib
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

request_path=Path(__file__).with_name('byte_requests.json')
request=json.loads(request_path.read_text())
result={'schema_version':1,'scope':'read-only exact source/raw/official/FNIT file audit and four actual-manifest schema fixtures; no MRI solver or GPU launched','host':socket.gethostname(),'started_UTC':datetime.now(timezone.utc).isoformat(),'python':sys.version,'numpy':np.__version__,'nibabel':nib.__version__,'audit_script_sha256':sha(__file__),'requests_sha256':sha(request_path),'file_checks':[],'geometry_checks':[],'errors':[]}
start=time.monotonic()
for row in request['requests']:
    p=Path(row['path'])
    try:
        digest=sha(p)
        record=dict(row,size_bytes=p.stat().st_size,actual_sha256=digest,matched=digest==row['sha256'])
        if not record['matched']:result['errors'].append({'path':str(p),'reason':'SHA mismatch'})
        result['file_checks'].append(record)
    except Exception as e:result['errors'].append({'path':str(p),'reason':str(e)})

for case in request['geometry_cases']:
    try:
        ap=nib.load(case['raw_AP']);pa=nib.load(case['raw_PA']);own=nib.load(case['official_packing']);actual=nib.load(case['actual_FNIT_packing'])
        ap_frame=np.asarray(ap.dataobj[...,case['indices'][0]])
        pa_frame=np.asarray(pa.dataobj[...,case['indices'][1]])
        own_array=np.asarray(own.dataobj);actual_array=np.asarray(actual.dataobj)
        assert own.shape==actual.shape==(96,96,60,2)
        assert np.array_equal(own.affine,actual.affine) and np.array_equal(own.affine,ap.affine)
        assert np.array_equal(own_array,actual_array)
        assert np.array_equal(own_array[...,0],ap_frame) and np.array_equal(own_array[...,1],pa_frame)
        data=nib.load(case['data']);mask=nib.load(case['mask']);bvecs=np.loadtxt(case['rotated_bvecs'])
        assert data.shape==tuple(case['expected_geometry']['shape'])==(96,96,60,102)
        assert np.array_equal(data.affine,np.asarray(case['expected_geometry']['affine']))
        assert mask.shape==data.shape[:3] and np.array_equal(mask.affine,data.affine)
        assert bvecs.shape==(3,102) and np.isfinite(bvecs).all()
        result['geometry_checks'].append({'subject':case['subject'],'AP_PA_indices':case['indices'],'full_packing_voxels_exact':True,'packing_first_AP_affine_exact':True,'corrected_shape':list(data.shape),'corrected_affine_exact':True,'mask_grid_exact':True,'rotated_bvecs_shape':list(bvecs.shape),'passed':True})
    except Exception as e:result['errors'].append({'subject':case['subject'],'reason':str(e)})

sys.path.insert(0,str(Path(__file__).parent/'origin_route_tools_v2'))
log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromName('test_origin_manifest_schema')
test=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
result['schema_tests']={'tests_run':test.testsRun,'failures':len(test.failures),'errors':len(test.errors),'successful':test.wasSuccessful(),'log':log.getvalue()}
result['elapsed_seconds']=time.monotonic()-start
result['completed_UTC']=datetime.now(timezone.utc).isoformat()
result['passed']=not result['errors'] and len(result['geometry_checks'])==6 and test.wasSuccessful() and test.testsRun==4
Path(__file__).with_name('audit_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':result['passed'],'file_checks':len(result['file_checks']),'geometry_checks':len(result['geometry_checks']),'schema_tests':result['schema_tests'],'errors':result['errors'],'elapsed_seconds':result['elapsed_seconds']},indent=2))
raise SystemExit(0 if result['passed'] else 1)
