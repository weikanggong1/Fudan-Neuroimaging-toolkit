"""Verify published pinned resources and actual installers using anonymous GETs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import time
from urllib.request import urlopen

from fnit._release_assets import RELEASE_BASE, release_url_for
from fnit.mshbm import assets_setup as mshbm
from fnit.recon_all import assets as recon


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_get(record):
    size = 0
    checksum = hashlib.sha256()
    with urlopen(RELEASE_BASE + record['name'], timeout=120) as response:
        assert response.status == 200
        for chunk in iter(lambda: response.read(1024 * 1024), b''):
            size += len(chunk)
            checksum.update(chunk)
    assert size == record['size'] and checksum.hexdigest() == record['sha256']
    return {'name': record['name'], 'size': size, 'sha256': checksum.hexdigest(),
            'full_anonymous_get': True, 'status': 'passed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmation', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    arguments = parser.parse_args()
    records = json.loads(arguments.confirmation.read_text())['resources']
    assert len(records) == 20
    start = time.perf_counter()
    report = {'status': 'running', 'scope': 'Resource installation only; no MRI benchmark',
              'authentication': 'anonymous; no credential headers', 'public_get': [],
              'installer_requests': []}
    def save():
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for job in as_completed([pool.submit(public_get, record) for record in records]):
            row = job.result()
            report['public_get'].append(row)
            save()
            print('GET verified', row['name'], flush=True)
    original_opener = urlopen
    def mirror_only(request, *positional, **keywords):
        url = request.full_url if hasattr(request, 'full_url') else request
        if not url.startswith(RELEASE_BASE):
            raise AssertionError('Author-source fallback was attempted')
        report['installer_requests'].append(url)
        return original_opener(request, *positional, **keywords)
    recon.urlopen = mirror_only
    mshbm.urlopen = mirror_only
    root = arguments.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    reconstruction_results = []
    for record in records:
        if record['group'] != 'recon':
            continue
        path = recon.download_asset(record['relative'], root / 'recon')
        assert path.stat().st_size == record['size'] and digest(path) == record['sha256']
        reconstruction_results.append(record['relative'])
        save()
        print('Installed', record['name'], flush=True)
    assert len(reconstruction_results) == 17
    projection = mshbm.prepare_projection_assets(root / 'mshbm', arguments.reference)
    for name in ('left_mni.surf.gii', 'right_mni.surf.gii', 'cortex_estimate.nii.gz'):
        _, size, checksum = mshbm.FILES[name]
        path = root / 'mshbm' / name
        assert path.stat().st_size == size and digest(path) == checksum
    import nibabel as nib
    import numpy as np
    reference = nib.load(arguments.reference)
    mask = nib.load(projection['cortical_mask'])
    assert mask.shape == reference.shape[:3]
    assert np.array_equal(mask.affine, reference.affine)
    assert mask.get_data_dtype() == np.dtype('uint8')
    assert mask.header.get_xyzt_units()[0] == reference.header.get_xyzt_units()[0]
    gm = next(record for record in records if record['group'] == 'oxford')
    gm_path = root / 'template_GM.nii.gz'
    gm_path.write_bytes(original_opener(RELEASE_BASE + gm['name'], timeout=120).read())
    assert gm_path.stat().st_size == gm['size'] and digest(gm_path) == gm['sha256']
    gm_image = nib.load(gm_path)
    report.update(status='passed', public_files_verified=20,
                  reconstruction_paths_verified=len(reconstruction_results),
                  mshbm_source_files_verified=3, author_fallbacks=0,
                  gm_release_fetch_verified=True,
                  gm_header={'shape': list(gm_image.shape), 'dtype': str(gm_image.get_data_dtype()),
                             'units': list(gm_image.header.get_xyzt_units())},
                  mask_header={'shape': list(mask.shape), 'dtype': str(mask.get_data_dtype()),
                               'units': list(mask.header.get_xyzt_units())},
                  installer_release_requests=len(report['installer_requests']),
                  elapsed_resource_validation_seconds=time.perf_counter() - start)
    assert report['installer_release_requests'] == 20
    save()
    print('Passed 20 new anonymous GETs, 17 recon paths, all 3 MS-HBM files, and GM fetch.', flush=True)


if __name__ == '__main__':
    main()
