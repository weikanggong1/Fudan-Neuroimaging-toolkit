"""隔离同输入未插桩参考重复；只报告已测几何范围，不影响生产。"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import nibabel as nib
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['case', 'previous', 'binary', 'assets', 'reference-license', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hemisphere', choices=['lh', 'rh'], required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    h = args.hemisphere
    names = [f'surf/{h}.white', f'surf/autodet.gw.stats.{h}.dat', f'label/{h}.cortex.label',
             f'label/{h}.cortex+hipamyg.label', *[f'mri/{name}.mgz' for name in ('brain.finalsurfs', 'wm', 'aseg.presurf')]]
    inputs = {name: sha(args.case / 'candidate' / name) for name in names}
    previous_report = json.loads((args.previous / 'report.json').read_text())
    if inputs != previous_report['input_sha256'] or sha(args.binary) != previous_report['binary_sha256']:
        raise ValueError('reference repeat must use the identical input and binary hashes')
    root = args.output / 'plain'
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.case / 'candidate' / name, target)
    surf, mri, labels = root / 'surf', root / 'mri', root / 'label'
    command = [str(args.binary), '--adgws-in', str(surf / f'autodet.gw.stats.{h}.dat'), '--seg', str(mri / 'aseg.presurf.mgz'),
               '--threads', str(args.threads), '--wm', str(mri / 'wm.mgz'), '--invol', str(mri / 'brain.finalsurfs.mgz'),
               f'--{h}', '--i', str(surf / f'{h}.white'), '--o', str(surf / f'{h}.pial.T1'), '--pial', '--nsmooth', '0',
               '--rip-label', str(labels / f'{h}.cortex+hipamyg.label'), '--pin-medial-wall', str(labels / f'{h}.cortex.label'),
               '--repulse-surf', str(surf / f'{h}.white'), '--white-surf', str(surf / f'{h}.white'), '--restore-255']
    env = dict(os.environ, FREESURFER_HOME=str(args.assets), SUBJECTS_DIR=str(root.parent), FS_LICENSE=str(args.reference_license))
    env.pop('PLACE_GRAD_PREFIX', None)
    env.pop('PLACE_VALUES_PREFIX', None)
    report = dict(status='running', scope='same-input two uninstrumented original-source runs; isolated reference only',
                  input_sha256=inputs, binary_sha256=sha(args.binary), script_sha256=sha(__file__), threads=args.threads,
                  previous_report_sha256=sha(args.previous / 'report.json'), thresholds_changed=False)
    def save():
        pending = args.output / 'report.pending'
        pending.write_text(json.dumps(report, indent=2) + '\n')
        pending.replace(args.output / 'report.json')
    save()
    start = time.perf_counter()
    with (args.output / 'reference.private.log').open('w') as log:
        process = subprocess.run(command, cwd=mri, env=env, stdout=log, stderr=subprocess.STDOUT)
    report.update(exit_code=process.returncode, full_command_wall_seconds=time.perf_counter() - start)
    save()
    if process.returncode:
        raise RuntimeError('reference repeat failed')
    first, faces = nib.freesurfer.read_geometry(str(args.previous / f'conda/surf/{h}.pial.T1'))
    second, second_faces = nib.freesurfer.read_geometry(str(root / f'surf/{h}.pial.T1'))
    if first.shape != second.shape:
        raise ValueError('reference repeat changed the vertex domain')
    difference = second - first
    report.update(ordered_faces_exact=bool(np.array_equal(faces, second_faces)),
                  coordinate_different_elements=int(np.count_nonzero(difference)),
                  coordinate_max_absolute_mm=float(np.abs(difference).max()),
                  coordinate_rms_mm=float(np.sqrt(np.mean(difference * difference))),
                  geometry_exact=bool(np.array_equal(first, second) and np.array_equal(faces, second_faces)),
                  repeat_surface_sha256=sha(root / f'surf/{h}.pial.T1'),
                  prior_surface_sha256=sha(args.previous / f'conda/surf/{h}.pial.T1'),
                  input_unchanged=inputs == {name: sha(args.case / 'candidate' / name) for name in names},
                  status='complete_reference_repeat')
    save()


if __name__ == '__main__':
    main()
