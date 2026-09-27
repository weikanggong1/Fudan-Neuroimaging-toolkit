# Isolated post-CC volume chain on the completed v2 subject

**Decision:** Keep this chain out of the default runner for now. On the current
approximate white/pial surfaces, it made the official final `aseg.mgz`
comparison worse: 104,986 unequal voxels before, **142,536 after**. The
existing Python ports have already matched the official commands on frozen
*official* inputs ([volmask](../../VOLMASK.md),
[relabel](../../RELABEL_HYPOINTENSITIES.md),
[fix-presurf](../../SURF2VOLSEG_FIX.md)). This experiment tests their combined
output using the completed v2 subject's different surfaces and labels.

## Inputs and isolation

The v2 E2E status was `exit=0` and its run report `complete` before the probe
started on headcw. The script copied the entire candidate subject to a new
scratch directory and changed only that copy. It checked SHA-256 for all seven
required source files before and after; `source_input_sha256_unchanged=true`.
The candidate was `reconall_conda_cpp_cc_e2e_20260927/subjects/sub01`; the
reference was `reconall_benchmark_pair_ac_20260924/official_subjects/a_official`.
Both derive from the same T1, SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`.
The full [machine report](report.json), [timing](time.txt), [exit status](status.txt),
[executed script](../probe_post_cc_chain.py), and [source hashes](source.sha256)
are archived here. The first scratch attempt computed the ribbon but failed to
serialize a NumPy integer in its JSON report ([traceback](initial_probe_error.log));
the corrected probe ran from a fresh scratch directory and exited 0. This did not edit the E2E subject.

## Voxel and format checks

All listed volumes are 256³. Each generated output has the reference affine
exactly. `aseg.presurf.mgz` entering this experiment already matched official
at **0/16,777,216 differing voxels**.

| Output from candidate inputs | Unequal voxels vs official | Candidate / official dtype | MGH header exact |
| --- | ---: | --- | --- |
| `ribbon.mgz` | 227,112 | uint8 / uint8 | yes |
| `lh.ribbon.mgz` | 105,802 | uint8 / uint8 | yes |
| `rh.ribbon.mgz` | 108,521 | uint8 / uint8 | yes |
| `aseg.presurf.hypos.mgz` | 879 | float32 / float32 | yes |
| final `aseg.mgz` | **142,536** | int32 / int32 | yes |

The candidate white surfaces made the relabel step change 878 presurf voxels;
the official white surfaces changed only one. The postprocessed candidate
`aseg.mgz` changed 102,472 voxels: 31,075 previous errors were corrected,
68,625 previously correct voxels became wrong, and 2,772 wrong voxels changed
to a different wrong value. The net error increased by 37,550 voxels.

The pinned FreeSurfer `mri_surf2volseg.cpp` allocates its output as `MRI_INT`.
The Python fix function currently retains its input float32 type, so this
probe separately saved the final integer labels as int32 with the source
geometry header. Its **entire MGH header** and dtype match the archived
official final aseg. The remaining difference is in voxel values; complete
decompressed MGH files therefore also differ.

## Runtime and integration boundary

| Isolated headcw stage | Seconds |
| --- | ---: |
| Copy whole subject | 1.03 |
| Python/Numba volmask | 10.04 |
| Python relabel hypointensities | 5.83 |
| Python fix presurf with ribbon | 10.09 |
| int32 final save | 0.60 |

The process took **34.33 s** including imports, copying, comparison, and JSON
writing (max RSS 866,852 KiB). These are unpaired to native commands on the
same candidate inputs. Earlier paired frozen official-input timings are in
the three stage reports linked above; they cannot establish a v2 end-to-end
speed gain.

A future production handoff would run `mris_volmask` equivalent output after
both white and pial surfaces, then `mri_relabel_hypointensities`, then
`mri_surf2volseg --fix-presurf-with-ribbon`, and reload the resulting int32
`aseg.mgz` before parcel projection and statistics. The current v2 runner
instead uses the earlier `aseg` array after surface generation. It also
produces no-GA `cortex.label`; official `label-cortex --fix-ga` additionally
requires `entowm.mgz`, which this candidate lacks. Since the white/pial meshes
and cortex labels are already different, swapping these short Python stages
for C++ would not resolve the observed final aseg error. Revisit this chain
after geometry and `entowm`/cortex-label inputs match, then compare each stage
on the **same candidate inputs** with the native command before enabling it.
