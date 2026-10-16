# Connected EntoWM stage: fixed T1

The Python/SimpleITK input chain produced `nu.mgz` from the fixed public
`sub-01_T1w.nii.gz`. The native-free PyTorch `mri_entowm_seg` consumed that file
with the separately stored `entowm.fsm31.t1.nstd00-30.nstd21-108.h5` and
`entowm.ctab` assets. The comparator was the archived, unmodified FreeSurfer
8.2 `a_official` subject. The fixed input SHA-256 was
`db518aae9ea0dee0c7420726bd5f1fd663ae19f48e7ef550111480a782475754`.
The archived official `nu.mgz` differs from this fresh Python `nu.mgz` in 34
of 16,777,216 voxels; this is therefore a connected archived-output comparison,
not a fresh native same-input EntoWM pair.

After retaining the input MGH header while writing int32 labels, all
16,777,216 output labels matched the archived official segmentation. All five
label Dice scores were 1.0, as were the affine, 284-byte MGH header, and voxel
payload. The MGZ files have different complete SHA-256 values because the
FreeSurfer output includes a color table and trailing metadata that this stage
does not serialize. Its label values and spatial metadata match.

The four `entowm.stats` label counts also match exactly. Posterior-based volumes
differ by at most 0.0227 mm³, at label 4006. The Talairach voxel LTA was
created from the Python affine LTA without a native utility. Its determinant
gives eTIV 1,310,267.038702 versus 1,310,266.552537 mm³ officially, an
absolute difference of 0.486165 mm³ (about 0.37 ppm). The subject CLI also
accepts the existing XFM directly when a voxel LTA is unavailable; that
fallback was independently exercised with a 1.450583 mm³ eTIV difference.
One control-host CPU call including inference and output write took 11.01 s under
shared load; it is not a paired native speed estimate. The full stats file is
not byte-identical because the eTIV and posterior volumes differ slightly.

The [machine-readable report](connected_entowm_cpu_20260926.json),
[comparator](experimental/compare_connected_entowm.py), and
[bounded runner](experimental/run_connected_entowm_cpu.sh) record this fixed
case. Other subjects, CUDA EntoWM inference, and final end-to-end recon-all
remain untested. No installed FreeSurfer executable was called by the
candidate stage.
