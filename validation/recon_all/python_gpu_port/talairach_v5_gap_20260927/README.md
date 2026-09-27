# MNI305 Talairach LTA first difference on the v5 real T1

This read-only audit uses the v5 FNIT prefix replay of
examples/data/sub-01_T1w.nii.gz (SHA-256
f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a)
and its completed unmodified FreeSurfer 8.2 reconstruction on gpucw1.
The [probe](probe.py) compares saved outputs; it runs no model and changes
neither subject. [report.json](report.json) records hashes and numbers.
The v5 subject was a downstream prefix replay, not a new completed
T1-to-all-outputs reconstruction.

| Boundary | Same-T1 v5 versus official |
| --- | ---: |
| synthstrip.mgz | 0/16,777,216 voxel differences; affine max difference 0 |
| transforms/synthmorph.mni305/aff.lta | Maximum 4×4 matrix-element difference 0.035349 |
| transforms/talairach.xfm.lta | Maximum 4×4 matrix-element difference 0.047333 |
| eTIV computed from the voxel LTA | FNIT 1,309,444.005577 versus official LTA 1,310,266.552537 mm³; difference -822.546959 mm³ |

The first measured difference is the MNI305 affine output, after an exact
SynthStrip image. The following voxel-LTA conversion carries and slightly
changes that difference. The separate
[venous-sinus statistics comparison](../mni_aux_connected_20260927/vsinus_stats_comparison.json)
shows an official stats eTIV of 1,310,266.467668 mm³, so its observed
-822.462091 mm³ error is mainly inherited from the Talairach LTA.
The remaining 0.084869 mm³ between the official-LTA formula and the
official stats value reflects an unisolated estimator/serialization path.

An [earlier CPU PyTorch SynthMorph replay](../INPUT_TALAIRACH_CHAIN.md)
on this real T1 produced an LTA differing from official by at most
0.00012255 matrix units and eTIV by 0.486165 mm³. Its process, device
and upstream snapshot differ from v5, so that observation suggests a
device-dependent numerical path but does not itself validate a production
fallback. The current SynthMorph constructor enables CUDA matmul and cuDNN
TF32; whether TF32 is the responsible operation is **unproven**. A
frozen-input, same-process GPU run with the relevant precision flags
toggled is needed to isolate it. Shared gpucw1 GPUs were occupied during
this audit, so no such run or production edit was made.

Existing v5 nu, brain, brainmask, entowm, aseg.presurf and other MRI-prefix
volumes remain voxel-exact despite this LTA gap on this T1. Switching the
Talairach registration device without revalidating those downstream files
would risk losing that established result. The white/pial and atlas gaps
remain larger than this 0.063% eTIV difference.
