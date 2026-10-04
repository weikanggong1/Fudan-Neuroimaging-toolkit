"""Plot brain-masked real-image outputs, excluding face and scalp voxels."""
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--latest-linear-runs', help='optional separate v3 rigid/affine CLI outputs')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.runs)
    mask_image = nib.load(args.mask)
    mask = np.asarray(mask_image.dataobj) > .5
    occupied = np.argwhere(mask)
    low = occupied.min(axis=0)
    high = occupied.max(axis=0)+1
    center = np.median(occupied, axis=0).astype(int)
    z = center[2]
    region = (slice(low[0], high[0]), slice(low[1], high[1]), z)
    figure, axes = plt.subplots(4, 3, figsize=(10, 12), constrained_layout=True)
    rows = []
    for row, model in enumerate(('rigid','affine','deform','joint')):
        reference_path = root/(model+'_1_reference')/'moved.nii.gz'
        candidate_path = (Path(args.latest_linear_runs)/(model+'_256_v3')/'moved.nii.gz'
                          if args.latest_linear_runs and model in ('rigid', 'affine') else
                          root/(model+'_2_candidate')/'moved.nii.gz')
        reference = nib.load(reference_path)
        candidate = nib.load(candidate_path)
        if candidate.shape != mask.shape or not np.allclose(reference.affine, mask_image.affine, atol=1e-3):
            raise ValueError('brain mask must match the complete output grid')
        r = np.where(mask, np.asarray(reference.dataobj), 0)
        c = np.where(mask, np.asarray(candidate.dataobj), 0)
        vmax = np.percentile(r[mask], 99)
        error = c-r
        error_scale = max(.01, np.percentile(np.abs(error[mask]), 99))
        for column, (data, title) in enumerate(((r,'Official CPU'),(c,'FNIT CPU'),(error,'FNIT - official'))):
            options = {'cmap':'gray','vmin':0,'vmax':vmax} if column<2 else {'cmap':'coolwarm','vmin':-error_scale,'vmax':error_scale}
            display = axes[row,column].imshow(data[region].T, origin='lower', **options)
            axes[row,column].set_title(model+' | '+title)
            axes[row,column].axis('off')
            if column == 2:figure.colorbar(display,ax=axes[row,column],fraction=.046,pad=.02)
        rows.append({'model':model,'reference_file':reference_path.name,'candidate_file':candidate_path.name,'brain_slice_z':int(z),'display_error_scale':float(error_scale),'brain_masked_before_display':True,'candidate_namespace':candidate_path.parent.name})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output,dpi=140)
    plt.close(figure)
    output.with_suffix('.json').write_text(json.dumps({'scope':'OpenNeuro ds003138 CC0, complete real-image registration outputs; axial display cropped to brain bounding box only; no cropped input benchmark', 'rows':rows,'figure_sha256':hashlib.sha256(output.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':
    main()
