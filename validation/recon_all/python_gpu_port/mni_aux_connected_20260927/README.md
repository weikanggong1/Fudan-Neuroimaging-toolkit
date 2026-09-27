# Connected MNI152 and auxiliary segmentation on a real T1

## Scope

This is a single-subject, same-T1, CPU validation of the candidate-generated
MNI152 LTA, MCA/dura and venous-sinus labels, and the downstream Python
brain.finalsurfs stage. It uses the deidentified
examples/data/sub-01_T1w.nii.gz scan (SHA-256
f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a)
and the completed FreeSurfer 8.2 reference on gpucw1. The v5 candidate's
seven reused MRI inputs (orig, nu, synthseg.rca, brain, brainmask, entowm,
aseg.presurf) each have 0 differing voxels and identical affine to official;
see [input_comparison.json](input_comparison.json). The candidate generates
its own crop, affine LTA and both auxiliary label maps. No official LTA or
label volume is passed into candidate inference. The saved official crop is
used only for comparison, since recon-all does not retain that intermediate
file in its final subject directory.

The Python stages are described in
[docs/recon_all/MNI_AUX_CHAIN.md](../../../../docs/recon_all/MNI_AUX_CHAIN.md).
The standalone finalsurfs stage is documented in
[docs/recon_all/FINAL_SURFS_CHAIN.md](../../../../docs/recon_all/FINAL_SURFS_CHAIN.md).
The production recon-all scheduler has not yet called the new MNI/aux
function; this is an isolated, connected stage test rather than a fresh
T1-to-all-outputs run.

## Input, output, and setup

The probe runs from the repository source tree with the FNIT Conda Python,
without invoking a FreeSurfer executable. It takes a candidate subject,
archived official subject, saved official crop for comparison only, external
weights and external assets. It writes the candidate's
mri/transforms/synthmorph.1.0mm.1.0mm/invol.crop.nii.gz, aff.lta and
reg.targ_to_invol.lta; mri/mca-dura.mgz; mri/vsinus.mgz;
stats/vsinus.stats; and mri/brain.finalsurfs.mgz with its manedit checkpoint.
It also writes [report.json](report.json) and [run.log](run.log).

The external weights are synthmorph.affine.2.h5,
mca-dura.both-lh.nstd21.fhs.h5 and vsinus.no-sp.m.all.nstd10-070.h5.
Three prior images and the cropped/full MNI152 1 mm images came from
hash-verified FreeSurfer 8.2 assets. The asset catalog includes four MNI152
registration files requiring an optional roughly 515 MB archive download;
this affine+aux API consumes two images, while the other two LTA templates
serve later registration stages. The source binaries and license were not
copied into the candidate package.

Reproduce on a host with the archived reference and inputs:

    PYTHONPATH=src python validation/recon_all/python_gpu_port/mni_aux_connected_20260927/probe.py \
      CANDIDATE_SUBJECT OFFICIAL_SUBJECT SAVED_OFFICIAL_CROP \
      WEIGHTS_DIR ASSETS_DIR report.json --device cpu

The saved raw report includes source and output SHA-256, 284-byte MGH
header checks, LTA distances, label counts and times. The linked
[probe.py](probe.py) calls the same production component functions that the
new wrapper uses. [compare_inputs.py](compare_inputs.py) generated the
upstream comparison. The one process used four CPU threads and finished
with exit code 0. A separate direct call of the module CLI on a fresh
subject directory also finished with exit code 0 in 35.68 s wall time and
4,922,680 KiB peak RSS. Its two label files and both LTA files were
byte-identical to the component probe outputs; see
[wrapper_comparison.json](wrapper_comparison.json) and
[wrapper.log](wrapper.log).

## Numerical result

| Same-T1 stage | Candidate vs FreeSurfer 8.2 |
| --- | ---: |
| Native T1 bounding crop | 0 differing voxels; 162×168×196; affine max absolute difference 7.63e-6 mm |
| Affine world matrix | Maximum absolute element difference 1.90e-5 |
| Full-MNI-to-native voxel LTA | Maximum absolute element difference 5.06e-5 voxel; 27-point mapped displacement mean 7.04e-5 mm, maximum 1.37e-4 mm |
| MCA/dura | 0/16,777,216 differing voxels; labels 6101/6102 each 531/590 voxels in both; float32 dtype, 284-byte MGH header and affine identical |
| Venous sinus | 0/16,777,216 differing voxels; all five nonzero label counts and intersections identical; float32 dtype, 284-byte MGH header and affine identical |
| brain.finalsurfs | 0/16,777,216 differing voxels; uint8 dtype, 284-byte MGH header and affine identical |

The [venous-sinus statistics comparison](vsinus_stats_comparison.json)
finds that all five rows match official numerically in Index, SegId,
NVoxels, Volume_mm3, Mean, StdDev, Min, Max and Range at the printed
precision. The stats files are not byte-identical: FNIT writes an 11-line
summary where FreeSurfer writes a 60-line metadata-rich report.
The eTIV measure also differs: FNIT 1,309,444.005577 versus official
1,310,266.467668 mm³, a difference of -822.462091 mm³ (-0.06277%).
The dominant source is the existing MNI305 Talairach LTA: applying FNIT's
eTIV formula to the candidate versus official LTA gives
1,309,444.005577 versus 1,310,266.552537 mm³. A further 0.084869 mm³
between the official-LTA formula and official stats has not been isolated.
Thus the auxiliary segmentation's regional statistics match, but its
eTIV and text format do not.

The compressed MGZ files have different SHA-256; the comparison establishes
voxel, grid and header equality, not byte identity. The previous isolated
GPU affine trial differed from official by 0.0685 voxel in the LTA. This CPU
run differs by 5.06e-5 voxel. The cause of the GPU numerical gap has not
been isolated, and the connected GPU path has not passed this test. All three
auxiliary model crop starts were previously measured equal under the larger
GPU LTA error; the present CPU run measured the final labels directly.

## Time and memory

| Stage | Candidate CPU wall time | Archived FreeSurfer wall time |
| --- | ---: | ---: |
| Bounding crop plus affine SynthMorph and LTA | 15.63 s | 1.86 s crop + 107.57 s mri_synthmorph |
| MCA/dura segmentation | 6.49 s | 118 s |
| Venous-sinus segmentation | 12.09 s | 55 s |
| Five finalsurfs edits | 8.58 s | 8.21 s isolated same-input replay |

The candidate stage sum is 42.79 s. The whole Python probe, including
imports and repeated 256³ comparisons, was 57.25 s wall time, 4,916,784 KiB
maximum resident set, on gpucw1. The official crop/SynthMorph and segmenter
times come from archived separate runs; load, process boundaries and date
differ. They are stage context, not a paired acceleration estimate. The
finalsurfs native timing was a separate same-input five-command replay.

The established result covers this T1 and CPU execution. The complete
recon-all scheduler still needs to connect these stages, generate surfaces
from the resulting finalsurfs, and pass the full output and vertex/ROI
comparison before end-to-end parity can be claimed.
