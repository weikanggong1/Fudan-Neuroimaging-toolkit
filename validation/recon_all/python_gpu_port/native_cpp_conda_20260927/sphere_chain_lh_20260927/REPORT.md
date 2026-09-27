# Real-T1 LH Python `sphere` → Python `sphere.reg` continuous-chain check

**Pass at the isolated stage boundary.** The fixed image is the repository's defaced derivative of real OpenNeuro ds000114 sub-01 T1w (published NIfTI SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`; frozen official `mri/orig.mgz` SHA-256 `7dde820d02968c9fe18056a9cc5cc776c1a6395c48dc4d3a47eb6ba9c399f518`). The Python-written conventional LH sphere from `run_standard_sphere` was passed directly as the `sphere` argument to the current-source `run_register_sphere`; no native sphere file or checkpoint was injected into registration. The companion `smoothwm`, `sulc`, and folding atlas were the frozen official files, so this is a two-stage surface-chain test, not a T1-to-reconstruction acceptance run.

## Exact inputs and commands

The validated Python sphere was generated on headcw with `PYTHONPATH=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/fnit_reconall_snapshot_20260926d/src`:

```bash
python -m fnit.recon_all.sphere_standard_run \
  /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/tessellate/sphere/sphere_next_20260926/full_native_complete/lh/surf/lh.inflated \
  /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/tessellate/sphere/sphere_next_20260926/full_native_complete/lh/surf/lh.smoothwm \
  /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/tessellate/sphere/sphere_next_20260926/next_gpu_probe_20260926/lh.sphere.api_headcw \
  --finish-device cpu --report /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/tessellate/sphere/sphere_next_20260926/next_gpu_probe_20260926/standard_sphere_lh_api_headcw.json
```

For the new connection test, the current local FNIT `src/fnit` was frozen into an isolated remote source directory; source archive SHA-256 `254d586a9aaf598db0a7734fb65ac22eec93c7dfe40970edc30c4abe449194b7`. The command was:

```bash
D=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/tessellate/sphere/sphere_next_20260926/sphere_register_chain_lh_20260927
B=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_main_20260924
W=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_python_gpu_20260925/tessellate/sphere/sphere_next_20260926
PYTHONPATH="$D/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m fnit.recon_all.mris_register_run \
  "$W/next_gpu_probe_20260926/lh.sphere.api_headcw" \
  "$B/single_subjects/fs_sub01/surf/lh.smoothwm" \
  "$B/single_subjects/fs_sub01/surf/lh.sulc" \
  "$B/bundle/average/lh.folding.atlas.acfb40.noaparc.i12.2016-08-02.tif" \
  "$D/lh.sphere.reg.python_chain" --overlap-device cpu \
  --report "$D/chain_lh_report.json"
```

`run_register_sphere(sphere, smoothwm, sulc, atlas, output, overlap_device='cpu')` reads those four ordered inputs and writes one FreeSurfer triangle surface. Its returned dict/JSON contains the paths, `overlap_device`, SHA-256 of each input and the output, SHA-256 of the temporary sulc seed, nested `sulc_pass` and `smoothwm_pass` reports (ordered updates, selected dt, next states and timings), and total seconds. The temporary seed is deleted after the call. Full input/output structure for both APIs is in the [current stage documentation](../../../../../docs/recon_all/CONDA_CPP_STAGES.md).

## Paired output and timing

| Check | Result |
| --- | --- |
| Python sphere file SHA-256 | `f6c5d1b2ccbe885952de33ee34d56898d860862bccf09422f0b3b2a8cdc2974c` |
| Fixed smoothwm / sulc / atlas SHA-256 | `36e199e4d971a39457a4ad1d0b8eed3a84feb236ada29c05e297ffdd865a5002` / `49801953491d37a6fac016bbd0342c5dc46d2524f1872416cee339469839481a` / `92f1dc820d778a67c143d4ea82597fd71a89ddc738dfa555b7ca32ed21e95c97` |
| Python registration output SHA-256 | `f303cd186995b088efc08fd0f2360c579dc6cf4ac2c4073b8f388dc41f2cf66e` |
| Archived official `lh.sphere.reg` SHA-256 | `801d345eb11c4b02e64aa453bc388f18e8cd47e8bb0beed7d75868d92276b8f8` |
| Ordered vertex payload | **106,622 / 106,622 exact**, 0 mm maximum error; payload SHA-256 on both sides `d62d63cb74d3270e43f85a5d5edd5905a4542d6f4577334eac38ffd76ffea686` |
| Ordered faces / volume geometry bytes | Both bitwise exact; SHA-256 `49603c29ba711ba3ae8a43ca39b1b929e4249ff7107615d1da5095baee49c99b` / `c4455e913b2688efdd569e9cb7822cf7e94c165139fe1878e9e52ef318056bd2` |
| Temporary sulc seed and Python final file versus the prior native-sphere-input Python run | Both whole-file SHA-256 identical; all 55 sulc and 51 smoothwm update schedules, chosen dt values and next states identical |
| Python stage times | Sphere 372.82 s including I/O; new registration 369.81 s inside API, 376.33 s shell wall including startup, peak RSS 766,384 KiB; two API times sum 742.63 s |
| Native observations on headcw | Same-input conventional sphere without snapshot writes 240.64 s; fresh native registration from official sphere 157.62 s; sum 398.26 s, measured at different times/load |

The whole-file Python/native `sphere.reg` SHA values differ because the writers use different creation stamps and FreeSurfer provenance tags. The ordered numerical surface and volume geometry bytes match. Python sphere→registration chaining is therefore verified for this LH frozen input. A matched controlled timing trial and an independent RH chained trial remain open. Upstream topology, white/pial, sulc generation and downstream cortical measurements in `native_free` are still not accepted as whole-T1 equivalent.

Machine-readable records: `CHAIN_LH_SUMMARY.json` (derived consistency checks), `chain_lh_sphere_report.json`, `chain_lh_report.json`, `chain_lh_geometry_audit.json`, `chain_lh_stderr.log`; the generated registered surface remains in the isolated remote directory shown above.
