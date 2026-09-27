# Real-T1 downstream replay after the matched MRI prefix

This is a **partial reconstruction**, not a fresh end-to-end run. The input is the
completed v3 MRI/stats prefix for `examples/data/sub-01_T1w.nii.gz` (SHA-256
`f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`).
The [replay script](replay.py) copied that prefix to an empty subject directory,
then ran both surface, sphere registration, annotation, projection and statistics
chains with the Python/Conda implementation on `gpucw1`, four CPU threads. It
neither reran preprocessing nor used official FreeSurfer output as a candidate
input. The archived official FreeSurfer 8.2 subject of the same T1 was read
only for comparisons. `run.json` records the SHA-256 of 11 copied MRI inputs;
`provenance.txt` records the executed script and Conda binary hashes. This
replay used the **earlier unpatched** Conda topology binary; the later
[exact bilateral GA/remesh result](../../../../../docs/recon_all/TOPOLOGY_CONDA_GA.md)
is a separate frozen-input validation and was not exercised here.

## Inputs, invocation and outputs

`replay.py prefix_subject subject_dir assets_dir native_bin_dir --device cpu
--threads 4` requires a finished FNIT MRI prefix, an empty output directory,
external atlas/template assets and the six source-built Conda C++ programs.
It returns no Python value. It writes a FreeSurfer-style `mri/`, `surf/`,
`label/`, `stats/`, `scripts/` subject tree plus
`fnit-prefix-replay-run.json`. The latter has `status`, `source_prefix`,
`subject_dir`, `device`, `threads`, `prefix_sha256`, ordered `stages` with wall
seconds, per-hemisphere surface/registration diagnostics and `total_seconds`.
The subject directory and medical images stay on the remote host; only the
JSON reports are committed. The harness also copied all v3 `mri/` and `stats/`
files; its 11 hashes do not cover every consumed file (notably
`aseg.auto.mgz`) or the external assets. The exact executed source archive is
identified by SHA-256 in `provenance.txt`, but is not pinned to a public Git
commit. Reproducing this snapshot requires that archived source and matching
external inputs.

```bash
python validation/recon_all/python_gpu_port/native_cpp_conda_20260927/prefix_surface_replay_20260927/replay.py \
  /path/to/completed-fnit-prefix /scratch/fnit-replay-subject \
  /path/to/assets /path/to/conda-source-build/bin --device cpu --threads 4
python validation/recon_all/python_gpu_port/compare_complete_subject.py \
  /path/to/official-subject /scratch/fnit-replay-subject \
  --report /scratch/strict_138.json
python validation/recon_all/python_gpu_port/native_cpp_conda_20260927/prefix_surface_replay_20260927/roi_summary.py \
  /path/to/official-subject /scratch/fnit-replay-subject \
  --report /scratch/roi_summary.json
```

The official full reference is `recon-all -i sub-01_T1w.nii.gz -s a_official
-sd /path/to/official_subjects -all -parallel -openmp 4 -itkthreads 1` under
FreeSurfer 8.2. It runs preprocessing as well as the downstream stages, so its
full runtime is not a matched runtime denominator for this replay.

## Accuracy

The fixed [138-file comparator](strict_138.json) passed **19/138** targets;
**47** official outputs were absent and **72** existed but failed their
type/geometry/numeric checks. The 19 passing targets are copied, previously
validated MRI-prefix files. None of the final surface, vertex-map, annotation
or statistics targets passed. `aseg.mgz` differs at **104,986 / 16,777,216**
voxels; `aparc+aseg.mgz` differs at **120,530** voxels. These voxel figures
must not be used as a cortical metric agreement claim. Compared with v3,
parcellation voxel counts improved modestly, while cortical thickness, area
and curvature region MAEs worsened; the replay is not a uniform improvement.

`roi_summary.json` compares every one of the 34 `aparc.stats` rows in each
hemisphere, and 45 `aseg.stats` rows. MAE is the unweighted mean of absolute
region differences; median relative error excludes zero reference values.

| Field | Left MAE; median relative | Right MAE; median relative |
| --- | ---: | ---: |
| Cortical `ThickAvg` | 0.737 mm; 29.72% | 0.799 mm; 35.24% |
| Cortical `SurfArea` | 634.6 mm²; 29.80% | 627.6 mm²; 30.45% |
| Cortical `GrayVol` | 2,523.4 mm³; 44.74% | 2,586.8 mm³; 51.06% |
| Cortical `MeanCurv` | 0.0585 1/mm; 45.07% | 0.0600 1/mm; 42.93% |

The 45 aseg volumes have MAE **507.65 mm³** and median absolute relative
error **0.170%** among nonzero reference regions. This summary hides a large
CSF difference: **+22,088.8 mm³**. The global `CortexVol` measure is
**353,949.84 → 497,264.49 mm³**, a **+40.49%** difference. The candidate
`orig` has 106,695/105,649 vertices (left/right), while the reference has
106,622/105,541. Vertex indices in this replay are therefore **not paired**;
the separate [exact-orig probe](../exact_orig_vertex_probe_20260927/README.md)
reports valid paired errors after exact topology.

## Time and interpretation

The replay exited 0 and completed 21 timed stages in **4,017.65 s** internally;
`/usr/bin/time` measured **1:07:02** process wall and 1,749,824 KiB peak
resident memory. The two surface stages took 1,188.37/1,031.96 s, and the two
sphere registrations took 733.25/721.55 s. These four stages account for
**3,675.13 s (91.47%)** of the internal total. All times are from a shared
host; nested timings in `run.json` must not be added to top-level stage times.
The archived official full reconstruction took 6,795 s, but ran a different
stage set. The replay's shorter time does not imply a speedup for equivalent
reconstruction.

The matched MRI prefix localizes the main cortical gap downstream. The later
frozen-input topology patch aligns both final `orig` meshes exactly, but the
[same-index vertex probe](../exact_orig_vertex_probe_20260927/README.md)
still found thickness MAE 1.080/1.103 mm: current final smoothing begins at
`orig` and white/pial placement is approximate. Official final
`smoothwm` begins at **`white.preaparc` with three smoothing passes**, whereas
this replay began at `orig` with ten. The [correct Python operator is exact on
the same input](../../../../../docs/recon_all/SMOOTHWM_FINAL_PARITY.md);
integrating it requires genuine `white.preaparc`, and subsequent pial,
registration, parcellation and final aseg checks. The newer exact-topology
source still needs a fresh continuous same-T1 subject comparison.

Raw evidence: [run stages](run.json), [strict files](strict_138.json),
[ROI rows](roi_summary.json), [wall/RSS](replay.time), [provenance](provenance.txt).
