#!/usr/bin/env python3
"""Plot actual successful official CPU b0/masks and stage timings; EDDY is excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-root', type=Path, required=True)
    parser.add_argument('--heldout-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--subjects', nargs='+', default=['CON01', 'CON03'] + [f'CON{i:02d}' for i in range(4,12)], help='actual successful CPU cases to display; default all ten')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects
    columns = len(subjects) if len(subjects) <= 5 else (len(subjects)+1)//2
    rows = (len(subjects)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(3*columns, 4*rows), facecolor='white', squeeze=False)
    timings, provenance = [], []
    for subject, axis in zip(subjects, axes.flat):
        root = args.pilot_root if subject in ('CON01', 'CON03') else args.heldout_root
        directory = root / f'sub-{subject}'
        report_path = directory / 'report.json'
        report = json.loads(report_path.read_text())
        commands = {x['stage']:x for x in report['commands']}
        assert all(commands[k]['returncode'] == 0 for k in ('official_topup','official_synthstrip_CPU'))
        paths = [directory/'mask/b0_mean.nii.gz', directory/'mask/nodif_brain_mask.nii.gz']
        brain, mask = [np.asarray(nib.load(p).dataobj) for p in paths]
        assert brain.shape == mask.shape and np.isfinite(brain).all() and np.isin(mask,(0,1)).all()
        z = brain.shape[2]//2
        axis.imshow(brain[:,:,z].T, origin='lower', cmap='gray', vmin=0, vmax=np.percentile(brain[mask>0],99))
        axis.contour(mask[:,:,z].T, levels=[.5], colors=['#ffbd59'], linewidths=.8)
        axis.set_title(f'{subject} | AP{report["selection"]["ap_index"]}/PA{report["selection"]["pa_index"]}', fontsize=10)
        axis.set_axis_off()
        timings.append([commands[k]['wall_seconds'] for k in ('official_topup','official_synthstrip_CPU')])
        provenance.append({'subject':subject, 'report':str(report_path),'report_sha256':digest(report_path),
                           'images_sha256':{str(p):digest(p) for p in paths},'native_axis2_slice':z,
                           'brain_shape':list(brain.shape),'timings_seconds':timings[-1],
                           'scope':'successful CPU TOPUP and SynthStrip only; no EDDY/full-chain claim'})
    for unused_axis in list(axes.flat)[len(subjects):]: unused_axis.set_axis_off()
    fig.suptitle(f'Official TOPUP mean b0 and SynthStrip masks | {len(subjects)} actual raw cases',fontsize=15)
    fig.text(.5,.015,'Native voxel planes (axis 2 midpoint), no display resampling. Orange: official mask. CPU stages only.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.93));fig.subplots_adjust(hspace=.2);fig.savefig(args.output_dir/'official_CPU_brain_masks.png',dpi=180);plt.close(fig)
    fig, axis=plt.subplots(figsize=(11,4.5));t=np.asarray(timings);x=np.arange(len(subjects))
    axis.bar(x,t[:,0],label='TOPUP',color='#4278ac');axis.bar(x,t[:,1],bottom=t[:,0],label='SynthStrip CPU',color='#ffbd59')
    axis.set_xticks(x,subjects);axis.set_ylabel('Measured command wall time (seconds)');axis.legend(frameon=False)
    axis.set_title('Official CPU stage times | gpucw1, 8 threads/case, shared host')
    axis.spines[['top','right']].set_visible(False)
    fig.text(.5,.015,'Stage sums exclude selection, ROI/merge, hashing, queue and EDDY; not end-to-end rawprep timing.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(args.output_dir/'official_CPU_stage_times.png',dpi=180);plt.close(fig)
    (args.output_dir/'official_CPU_figures.provenance.json').write_text(json.dumps({'subjects':provenance,'tool_sha256':digest(Path(__file__))},indent=2)+'\n')

if __name__ == '__main__': main()
