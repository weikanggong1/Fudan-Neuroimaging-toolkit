# White preaparc floating-point ablations and v5 placement gate

## Scope and exact inputs

All numerical runs use the real `examples/data/sub-01_T1w.nii.gz` scan (SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`) and the archived FreeSurfer 8.2 reference. The fixed-input ablations use the **same official** `lh.orig`, `brain.finalsurfs.mgz`, `wm.mgz`, `aseg.presurf.mgz`, and `autodet.gw.stats.lh.dat`. Ordered `lh.orig` has 106,622 vertices and 213,240 faces. The exact placement command, from the subject's `mri` directory, is:

```bash
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --wm wm.mgz --threads 4 --invol brain.finalsurfs.mgz --lh --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white --seg aseg.presurf.mgz --restore-255 --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 --outvol mrisps.wpa.mgz --debug-vertex 38359
```

Each run writes a FreeSurfer surface (`V × 3` float coordinates, `F × 3` ordered face indices, volume metadata), a 256³ diagnostic MGZ, a full debug log with selected vertex 38359 and 42 accepted steps, a process-time text file, and an exit-code file. The isolated `build.py` reads the unchanged Conda FreeSurfer source/build tree and writes two standalone binaries, a build manifest with exact compiler/linker arguments and hashes, and compile/link logs. `compare.py` reads the surfaces/debug logs and writes `comparison.json` with paired vertex displacement, first printed force/SSE mismatch, diagnostic volume hashes, and timings. These are diagnostic tools and do not change the production binary or repository.

## Compiler and library evidence

The installed binary contains GCC 4.8.5 in `.comment`; the Conda binary contains GCC 11.4.0 object code alongside GCC 4.8.5 library objects. Its force object `utils/mrisurf_compute_dxyz.cpp.o` is compiled with `-march=nocona -mtune=haswell -ftree-vectorize -O3`. Neither installed executable nor this Conda object contains FMA instructions in disassembly, so `-ffp-contract=off` cannot affect this candidate's active instructions. Both executables import `tanh` rather than `tanhf` and use the same `/lib64/libm.so.6` on the test host. They usually load different OpenMP runtimes: system `/lib64/libgomp.so.1` for the installed binary and Conda's `libgomp.so.1` for the source-built binary.

Two **single-object** builds copied the unchanged `libutils.a`, replaced only `mrisurf_compute_dxyz.cpp.o`, and linked a private executable using the original main object and other libraries. Variant A appends `-fno-tree-vectorize`; variant B appends `-O2` after the original `-O3`, preserving explicit `-ftree-vectorize`. Their changed object and executable SHA-256 values are in `manifest.json`; `ar t` verifies exactly one target object member in each archive. A third run used the unchanged Conda executable with `LD_PRELOAD=/lib64/libgomp.so.1`; `ldd` confirmed system libgomp resolution. All runs used four threads and the same authorized external license.

## Same-input results

| Candidate | First selected force difference after accepted step 5 | First global SSE difference at step 7 | Mean / P99 / max displacement versus installed | Vertices >0.1 mm | Vertices identical to original Conda |
| --- | --- | --- | --- | ---: | ---: |
| Original Conda | `dxyz.z −0.0303281` vs official `−0.030328` | 543343.9 vs official 543344.0 | 0.000420574 / 0.00699896 / 0.630004 mm | 50 | 106,622 |
| Force object `-fno-tree-vectorize` | Same | Same | Same | 50 | **106,622 / 106,622** |
| Force object `-O2` | Same | Same | Same | 50 | **106,622 / 106,622** |
| System `libgomp.so.1` preload | Same | Same | Same | 50 | **106,622 / 106,622** |

All four cases have identical ordered faces. Each diagnostic `mrisps.wpa.mgz` has the same SHA-256 as the installed result (`6b372d531de6be4743b88fe0df99c5249051ad46a7089beeb44b9e7adeea94af`). The exact coordinate match to the original Conda result shows that these toggles change neither the first visible discrepancy nor its amplified final geometry. There is **no justified production compiler flag or OpenMP runtime patch** from these tests. A different source object, a subtler hidden-precision input, or another instruction/code path remains possible; this run does not localize the cause to a single expression.

Debug-run process wall times were 299.98 s (`-fno-tree-vectorize`), 269.52 s (`-O2`), and 174.68 s (system libgomp preload). The archived original Conda debug run took 303.76 s; the archived official debug run took 223.44 s. These were separate shared-node observations with varying load, and the isolated flags did not improve accuracy. No speed claim follows from those times.

## Frozen v5 left white.preaparc placement

The separate v5 prefix replay used the same real T1. Its left `orig.premesh`/`orig` exist, but `orig` contains **106,695 vertices and 213,386 faces**, while the official mesh contains **106,622 and 213,240**. Native placement preserves topology, so vertex-by-vertex comparison with the official output is invalid. The frozen isolated subject copied v5 `lh.orig`, `lh.orig.premesh`, `wm.mgz`, and `aseg.presurf.mgz` with byte-identical hashes; their exact paths and hashes are in `v5_input_manifest.json`. It **borrowed the official** `brain.finalsurfs.mgz` (SHA-256 `c239ba0c807bd381662d388e198715da44103f0c724b3877a7c9623d4ccff9e7`). Python `write_autodet_stats` generated the LH threshold file from these inputs byte-identically to the official file (SHA-256 `10b313e9f29a9d6572f0aef4546d307f6514eaae44896fc19b4212891719c7d9`).

From the isolated subject's `mri` directory, the Conda source-built command was:

```bash
mris_place_surface --adgws-in ../surf/autodet.gw.stats.lh.dat --wm wm.mgz --threads 4 --invol brain.finalsurfs.mgz --lh --i ../surf/lh.orig --o ../surf/lh.white.preaparc --white --seg aseg.presurf.mgz --restore-255 --nsmooth 5 --rip-bg-no-annot --rip-bg --rip-bg-lof --restore-255 --outvol mrisps.wpa.mgz
```

It exited 0 in **284.88 s** wall time at four threads, with 1,120,880 KiB peak RSS. Inputs were a surface (`V × 3` coordinates, `F × 3` ordered faces), three 256³ MGZ volumes, and a text threshold file. Outputs were `surf/lh.white.preaparc` (106,695 ordered vertices; 213,386 faces), `mri/mrisps.wpa.mgz` (256³ MGZ), a process log, time file and exit code. The outvol SHA-256 `6b372d531de6be4743b88fe0df99c5249051ad46a7089beeb44b9e7adeea94af` matches the official outvol. The placed surface SHA-256 is `1938efb0f8d49dd11f04445cedc029e15e71787819f594d964ebd2834daadb1f`.

| Comparison, nearest vertices only | V5 approximate | Native placement on frozen v5 inputs |
| --- | ---: | ---: |
| Official → candidate mean / P99 | 0.778 / 3.079 mm | **0.126 / 0.543 mm** |
| Candidate → official mean / P99 | 0.579 / 1.468 mm | **0.126 / 0.543 mm** |

The paired movement from v5 approximate to native placed geometry averaged 0.800 mm. The nearest-vertex distances measure geometric proximity across unequal topologies; they do not test homologous vertices, thickness, area, curvature, cortical labels or complete reconstruction. `v5_lh_comparison.json` holds full distributions. The borrowed official intensity volume means this is a **frozen-input feasibility test**, not independent end-to-end evidence.

## Missing connected finalsurfs inputs

The v5 replay's configured external directories are `work/reconall_wm_agent_20260927/v3_weights` and `work/reconall_wm_agent_20260927/v3_assets`. These two model files and seven templates are all absent there. Paths are relative to their respective directories; all sizes and SHA-256 values come from the repository's verified catalogs and are also in `missing_assets.json`.

| Type | Relative path | Bytes | SHA-256 | Present and verified in v5 |
| --- | --- | ---: | --- | --- |
| `model` | `mca-dura.both-lh.nstd21.fhs.h5` | 3,294,856 | `da6a7b994e3e804cc3dc0e98e965c28a802ddcd38fd9b5c680d75cef285657b0` | False |
| `model` | `vsinus.no-sp.m.all.nstd10-070.h5` | 3,296,904 | `3d78948741306a31337468c86be55821913edb73855116fcb063b61135b90f12` | False |
| `template` | `average/mca-dura.prior.warp.mni152.1.0mm.lh.nii.gz` | 41,792 | `0ea9f9ffdcbc38b139e5ca374dc322cb60773c355ffbccf21efd812d8c0ee458` | False |
| `template` | `average/mca-dura.prior.warp.mni152.1.0mm.rh.nii.gz` | 41,927 | `5772a838e2fdd2ecf0c83de27bc572532cd8b9b8a1b00d919a385aab34e5710e` | False |
| `template` | `average/vsinus.no-sp.prior.mni152.1.0mm.mgz` | 269,881 | `b46661dda5cdde2a9c43cd2bad7ca096bd96bf2ba6c60f784d54cd8a75485126` | False |
| `template` | `average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.cropped.nii.gz` | 7,745,204 | `ef89b7aa615dcb4a68a1cad725a9b99314227f9777ed86f42aae61e4e89da19a` | False |
| `template` | `average/mni_icbm152_nlin_asym_09c/reg-targets/mni152.1.0mm.nii.gz` | 17,977,963 | `e4e1a25b66fef6b2cde8e4916d4a03d4369de6f24b0d3d4736baffc3d4a8c575` | False |
| `template` | `average/mni_icbm152_nlin_asym_09c/reg-targets/reg.1.0mm.to.1.0mm.cropped.lta` | 1,620 | `2715856af855b0308350bbe32f8f932e3432d4c96acbbe107eee9dbb1c2b2e07` | False |
| `template` | `average/mni_icbm152_nlin_asym_09c/reg-targets/reg.1.0mm.cropped.to.1.0mm.lta` | 1,620 | `2a80b62a9baceaa8c4312bf46d5468353b10ed2990e1704028907700cb13d790` | False |

Total extracted size is **32,671,767 bytes**. The existing `fnit.weights.WEIGHT_FILES` catalog has stable upstream v8.2.0 URLs and SHA-256 for both models, and `download_file(name, dest)` can acquire and verify them. The downloader-group update adds both models to `--model recon-all` and the three small priors to default `CORE_ASSETS`. The existing `fnit.recon_all.assets.ASSET_FILES` catalog covers all seven templates; the four MNI152 files remain available through explicit `--asset PATH` or `--all`. They are extracted from a **514,649,342-byte verified archive** by the current downloader, so they are excluded from default download while the subject-specific MNI152 registration stage is not connected. No new unverified URL was introduced in this report. An independent live download on headcw fetched both models and the three small priors (6,945,360 bytes total); each matched the catalog size and SHA-256 (`download_probe.json`). Direct per-file annex URLs for the four MNI152 members were **not verified**: one candidate returned HTTP 404 and three HEAD requests timed out. An alternative small-file source remains to be established before adding these four to default setup.

The subject-specific `mri/transforms/synthmorph.1.0mm.1.0mm/reg.targ_to_invol.lta` is **not downloadable**. It must be generated from the current T1 and MNI152 targets by an affine SynthMorph registration and transform composition before Python `mri_mcadura_seg` and `mri_vsinus_seg` can operate. The affine checkpoint `synthmorph.affine.2.h5` already exists in v5 weights (catalog size 51,455,312 bytes). The current reconstruction runner does not create this MNI152 LTA or call those auxiliary segmenters. It also has not connected the official `brain.mgz` mask sequence against MCA/dura and venous-sinus segmentations, followed by EntoWM and ACJ edits, into `brain.finalsurfs.mgz`. Replacing that file with `brain.mgz` would alter 23,792 of 16,777,216 voxels on this subject, with maximum intensity difference 228. No geometry placement should be wired into production with the borrowed official volume.

After connected `brain.finalsurfs.mgz` and topology parity are achieved, generate autodet thresholds from the actual `orig.premesh`, run white.preaparc from `orig`, generate the early cortex and hippocampus/amygdala labels, register/annotate sphere, run final white, then pial.T1 and downstream maps and statistics. The current runner computes white/pial approximations earlier; a single executable substitution would not reproduce the official stage dependencies.

## Reproducibility and limits

The real scan is `examples/data/sub-01_T1w.nii.gz` (SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`). Reference FreeSurfer 8.2 subject: `work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official`. Isolated remote diagnostic directory: `work/white_float_ablation_codex_20260927/`; source-built Conda binary: `work/reconall_cpp_conda_20260927/fs_cpp/bin/mris_place_surface`. `manifest.json` records exact source/object/binary hashes and compiler/linker commands; `comparison.json` records same-input ablations. No private license bytes, T1 image, native binaries, or official subject image are committed. The three debug wall-time observations were on a shared node at different times, and no equivalent speed claim is supported. This report proposes **no compiler, runtime, or geometry integration patch**.
