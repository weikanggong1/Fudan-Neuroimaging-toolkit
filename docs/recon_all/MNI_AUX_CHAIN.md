# MNI152 affine and auxiliary segmentations

The fnit.recon_all.mni_aux_chain module produces the subject-specific
transform and two label volumes needed by the Python brain.finalsurfs stage.
It calls PyTorch SynthMorph and the existing PyTorch MCA/dura and venous-sinus
segmenters. It does not invoke a FreeSurfer executable.

## Inputs and outputs

| Python function | Inputs | Outputs |
| --- | --- | --- |
| register_mni152_affine(subject_dir, weights_dir, assets_dir, device="cpu", threads=4) | subject_dir/mri/orig.mgz; external synthmorph.affine.2.h5; cropped and full 1 mm MNI152 NIfTI templates under assets_dir/average/mni_icbm152_nlin_asym_09c/reg-targets/ | mri/transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz, aff.lta, reg.targ_to_invol.lta; returns the last path. The final type-0 4×4 LTA maps full-MNI152 voxels to native voxels and records both volume geometries. |
| run_mni_aux_chain(subject_dir, weights_dir, assets_dir, device="cpu", threads=4) | Registration inputs; subject_dir/mri/nu.mgz and synthseg.rca.mgz; MCA/dura and venous-sinus H5 weights; three priors under assets_dir/average/ | Registration outputs; mri/mca-dura.mgz (labels 0, 6101, 6102), mri/vsinus.mgz (labels 0, 6111, 6112, 6115, 6116, 6117); stats/vsinus.stats. Returns a dict keyed lta, mca_dura, vsinus with Path values. Both label volumes retain the conformed nu.mgz grid and float32 MGH metadata. |

The input crop uses the nonzero bounding box of the conformed orig.mgz
with a three-voxel margin. SynthMorph estimates a cropped-native to
cropped-MNI152 world transform. Matrix composition converts the full MNI152
target voxels to native voxels for prior resampling. The auxiliary models
read this LTA, crop nu.mgz from the registered prior, run PyTorch inference,
and paste hard labels back on the subject grid. The venous-sinus function
clears labels where synthseg.rca.mgz marks cortex (3, 42).

Python API:

    from fnit.recon_all.mni_aux_chain import run_mni_aux_chain

    paths = run_mni_aux_chain(
        "/path/to/subjects/sub01", "/path/to/weights", "/path/to/assets",
        device="cuda:0", threads=4,
    )

Command line:

    python -m fnit.recon_all.mni_aux_chain /path/to/subjects/sub01 \
      --weights /path/to/weights --assets /path/to/assets \
      --device cuda:0 --threads 4

The generated stats/vsinus.stats has exact printed numeric rows for
the five venous-sinus regions on the tested T1, but its eTIV is derived from
the existing Talairach LTA and differs by 822.462091 mm³ from the official
reference. Its text metadata and spacing also differ; see the linked
comparison below.

The external asset catalog verifies MNI152 template and prior hashes. The
two MNI152 image files are optional downloads; the current asset installer
extracts them from a roughly 515 MB upstream archive. Models and templates
are not part of the Python package.

## Corresponding FreeSurfer 8.2 commands

    mri_mask -bb 3 orig.mgz orig.mgz invol.crop.nii.gz
    mri_synthmorph -m affine -t aff.lta invol.crop.nii.gz \
      mni152.1.0mm.cropped.nii.gz -j 4
    mri_concatenate_lta -invert1 -invertout aff.lta \
      reg.crop-to-invol.lta reg.invol_to_croptarg.lta
    mri_concatenate_lta -invert2 reg.1.0mm.to.1.0mm.cropped.lta \
      reg.invol_to_croptarg.lta reg.targ_to_invol.lta
    mri_mcadura_seg --i nu.mgz --o mca-dura.mgz --threads 4 \
      --synthmorphdir transforms/synthmorph.1.0mm.1.0mm
    mri_vsinus_seg --s sub01 --rca-synthseg --threads 4 \
      --synthmorphdir transforms/synthmorph.1.0mm.1.0mm

The [real-T1 comparison](../../validation/recon_all/python_gpu_port/mni_aux_connected_20260927/README.md)
reports the candidate LTA, both label maps, and the downstream
brain.finalsurfs.mgz against the saved FreeSurfer reconstruction. This
module is a callable stage; the whole-subject recon-all scheduler still
requires integration and a fresh end-to-end parity run.
