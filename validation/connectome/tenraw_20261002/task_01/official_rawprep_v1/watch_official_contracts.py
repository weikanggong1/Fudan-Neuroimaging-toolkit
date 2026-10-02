#!/usr/bin/env python3
"""Publish only completed official cases and activate frozen heldout GPU phase.

No GPU initialization or scientific computation occurs in this watcher.
Existing command/failure reports are read only; publication errors are separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import nibabel as nib
import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''): value.update(data)
    return value.hexdigest()


def save(path, value):
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, indent=2) + '\n'); partial.replace(path)


def publish(case, directory):
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text())
    if not report.get('completed'): return False
    eddy_stage = 'official_EDDY_CPU' if report.get('EDDY_solver') == 'cpu' else 'official_EDDY_GPU'
    expected_stages = ('official_topup', 'official_synthstrip_CPU', eddy_stage)
    commands = {record['stage']: record for record in report['commands']}
    if not all(commands[stage].get('returncode') == 0 for stage in expected_stages):
        raise ValueError('completed flag without all official successful commands')
    if commands[eddy_stage].get('budget_exceeded') or commands[eddy_stage].get('wrong_GPU_UUID'):
        raise ValueError('GPU budget/UUID failure')
    command = commands[eddy_stage]['command']
    if f'--topup={directory}/topup/fieldmap_out' not in command or f'--mask={directory}/mask/nodif_brain_mask.nii.gz' not in command:
        raise ValueError('official EDDY did not consume its own new field/mask')
    for path, digest in report['input_sha256'].items():
        if sha(path) != digest: raise ValueError('canonical raw changed')
    for path, digest in report['EDDY_inputs_sha256'].items():
        if sha(path) != digest: raise ValueError('own official EDDY dependency changed')
    for relative, digest in report['output_sha256'].items():
        if sha(directory / relative) != digest: raise ValueError('official output changed')
    lineage = report.get('source_CPU_stage_lineage')
    if lineage:
        if sha(lineage['report']) != lineage['report_sha256']:raise ValueError('source CPU report changed')
        for path,digest in lineage['source_EDDY_inputs_sha256'].items():
            if sha(path) != digest:raise ValueError('source official CPU dependency changed')
        for path,digest in report['copied_successful_CPU_output_sha256'].items():
            if sha(directory/path) != digest:raise ValueError('restored official CPU output changed')
    raw_image = nib.load(directory / 'raw/AP.nii.gz')
    corrected = nib.load(directory / 'eddy/data.nii.gz')
    mask_image = nib.load(directory / 'mask/nodif_brain_mask.nii.gz')
    if corrected.shape != raw_image.shape or mask_image.shape != raw_image.shape[:3]:
        raise ValueError('grid/frame count mismatch')
    if not np.allclose(corrected.affine, raw_image.affine, rtol=0, atol=1e-5) or not np.allclose(mask_image.affine, raw_image.affine, rtol=0, atol=1e-5):
        raise ValueError('native affine mismatch')
    bvals = np.loadtxt(directory / 'raw/AP.bval').reshape(-1)
    bvecs = np.loadtxt(directory / 'eddy/data.eddy_rotated_bvecs')
    if bvecs.shape != (3, len(bvals)) or len(bvals) != corrected.shape[3]: raise ValueError('gradient count mismatch')
    contract = {'completed': True, 'subject': case['subject'].removeprefix('sub-'), 'session': case['session'],
                'source': 'new canonical raw -> own official TOPUP -> own official SynthStrip -> own official EDDY',
                'data': str(directory / 'eddy/data.nii.gz'), 'mask': str(directory / 'mask/nodif_brain_mask.nii.gz'),
                'bvecs': str(directory / 'eddy/data.eddy_rotated_bvecs'), 'rotated_bvecs': str(directory / 'eddy/data.eddy_rotated_bvecs'),
                'bvals': str(directory / 'raw/AP.bval'), 'T1w': case['t1w'],
                'topup_prefix': str(directory / 'topup/fieldmap_out'), 'GPU_UUID': report['GPU_UUID'],
                'GP_seed': report['gp_seed'], 'ref_scan_no': report['selection']['ap_index'], 'EDDY_solver': report.get('EDDY_solver', 'gpu'),
                'report': str(report_path), 'report_sha256': sha(report_path),
                'raw_sha256': report['input_sha256'], 'output_sha256': report['output_sha256'],
                'shape': list(corrected.shape), 'verified_utc': datetime.now(timezone.utc).isoformat()}
    if lineage:
        contract['source_CPU_stage_lineage'] = lineage
        contract['timing_policy'] = report['timing_policy']
        contract['source'] = 'canonical raw -> restored SHA-verified own successful official TOPUP/SynthStrip -> new official CPU8 EDDY'
    target = directory / 'completed_contract_verified.json'
    if not target.exists(): save(target, contract)
    # Initial pilot tool did not emit a contract; add one without rewriting old evidence.
    if not (directory / 'completed_contract.json').exists(): save(directory / 'completed_contract.json', contract)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--heldout-root', type=Path, required=True)
    parser.add_argument('--reference-tool', type=Path, required=True)
    parser.add_argument('--fnit-source', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--publish-only', action='store_true', help='Validate and publish existing producer outputs; never dispatch another solver')
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())['cases']
    heldout = [f'CON{index:02d}' for index in range(4, 12)]
    published = set()
    activated = False
    proc = None
    log = None
    while True:
        status = {'scope': 'official rawprep only, no claim of downstream/full connectome completion',
                  'published_completed_cases': sorted(published), 'GPU_activated': activated,
                  'subjects': {}, 'observed_utc': datetime.now(timezone.utc).isoformat()}
        failures = False
        for case in cases:
            subject = case['subject'].removeprefix('sub-')
            root = args.pilot_root if subject in ('CON01', 'CON03') else args.heldout_root
            directory = root / f'sub-{subject}'
            report_path = directory / 'report.json'
            if not report_path.exists():
                status['subjects'][subject] = {'state': 'not_started'}; continue
            report = json.loads(report_path.read_text())
            status['subjects'][subject] = {'state': report['state'], 'completed': report['completed'],
                                           'CPU_prepared': report.get('CPU_prepared', False)}
            if report.get('state') == 'failed': failures = True
            try:
                if subject not in published and publish(case, directory): published.add(subject)
            except Exception as error:
                status['subjects'][subject]['contract_error'] = repr(error); failures = True
        status['published_completed_cases'] = sorted(published)
        status['all_ten_rawprep_completed'] = len(published) == 10
        summary = args.heldout_root / 'summary_cpu.json'
        cpu_ready = summary.exists() and json.loads(summary.read_text()).get('all_CPU_prepared', False)
        if not args.publish_only and not activated and cpu_ready and {'CON01', 'CON03'} <= published and not failures:
            command = [os.sys.executable, str(args.reference_tool), '--manifest', str(args.manifest),
                       '--raw-root', str(args.raw_root), '--fnit-source', str(args.fnit_source),
                       '--output-root', str(args.heldout_root), '--subjects', *heldout,
                       '--workers', '2', '--phase', 'gpu']
            log = (args.heldout_root.parent / 'official_rawprep_heldout_v1_GPU.log').open('w')
            proc = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            activated = True
            save(args.heldout_root / 'GPU_dispatch.json', {'pid': proc.pid, 'command': command,
                                                          'tool_sha256': sha(args.reference_tool),
                                                          'CPU_ready': True, 'both_pilots_completed': True})
        if proc is not None:
            status['GPU_controller_pid'] = proc.pid; status['GPU_controller_returncode'] = proc.poll()
        status['GPU_activated'] = activated
        save(args.status, status)
        if len(published) == 10: break
        if failures or (proc is not None and proc.poll() is not None and len(published) != 10):
            status['stopped_on_failure_preserved'] = True; save(args.status, status); break
        time.sleep(15)
    if log: log.close()


if __name__ == '__main__': main()
