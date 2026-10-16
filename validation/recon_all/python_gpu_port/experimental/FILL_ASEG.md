# Aseg-guided `mri_fill` port: fs_sub01

## Scope and source

The actual `recon-all` command is `mri_fill -a scripts/ponscc.cut.log -xform transforms/talairach.lta -segmentation aseg.presurf.mgz -ctab SubCorticalMassLUT.txt wm.mgz filled.mgz`. `-a` writes cutting-plane coordinates. The segmentation branch is `fill_with_aseg` in FreeSurfer 8.2.0 source commit `d932c45b7941662ea380a05efef580568b98d41a` (`mri_fill/mri_fill.cpp` SHA-256 `8f93f66aae7f1391a3314f3728a4301663d5c013a11bd9f9bdccc791ee0c544b`). It calls the UCHAR 26-neighbor Voronoi routine, 18-connected largest-component routine, and 6-connected hole removal. CC labels 251–255 are first replaced with left or right WM by signed fast-marching distance. Before this branch, the Talairach cutting plane can remove special WM values 200, 210, 220, 230, and 240. The output embeds the supplied color table.

`src/fnit/recon_all/fill_aseg_python.py` implements CC replacement, the aseg-guided fill, edited-on voxel voting, component selection, and hole removal with Python, NumPy, Numba, and SciPy. Its optional `cc_cut_mask` represents the preceding cutting-plane result. `src/fnit/recon_all/fill_cutting_plane_python.py` now computes that mask from the LTA, WM, and aseg, and writes the output MGZ with the supplied FreeSurfer color table. The complete aseg-guided `mri_fill` stage does not read a FreeSurfer executable or runtime package. It has not been added to the main recon-all entry point.

## Fixed inputs and references

All paths below are under `/data/workspaces/freesurfer_synth/work`. The official subject is `reconall_main_20260924/single_subjects/fs_sub01/mri`; isolated outputs are in `reconall_python_gpu_20260925/fill_ref`.

| File | SHA-256 |
| --- | --- |
| `wm.mgz` | `b190b1d8f9a4bf114c8a6623fdfd37cde0950de8f991b35243bccea46498ed5e` |
| `aseg.presurf.mgz` | `263653b66a9aacba4d0d704781c6e67e581c86a9f99374e37d7e3f28d1674ce2` |
| Existing official `filled.mgz` | `15848b056548ba8215c926c6ef70509b8f4859e767b36ec2d38af91e8fa3a2a0` |
| Isolated official `cc.mgz` cutting mask | `52695fba8b358d84e7bd86c18e7e91cc625077d7a03db67fe384bdfd1e233f7b` |
| Experimental CC-preclassified `aseg_edt.mgz` | `4e2b48c348a87f579d6d2fde8e2204ed44bd2a7b04c955ace6750fc706333f77` |
| Isolated official cutting mask for that experimental aseg | `ad8450e5694d4ebaad6c0355197af2d2c149f8816e16e4292fe0bbe437277414` |

The original segmentation has 2,263 CC voxels. The native cutting mask intersects five special WM marker voxels. Using the original input, an isolated native rerun differed from the existing `filled.mgz` at **0/16,777,216 voxels**. The copied native runtime was used only to create references and time the official command.

## Numerical gates

| Comparison | Mismatched voxels | Algorithm time on control-host |
| --- | ---: | ---: |
| Python, experimental preclassified aseg and its matching native cutting mask, versus native on those same inputs | **0 / 16,777,216** | 4.69 s |
| Python, original aseg and original native cutting mask, versus original native output | **0 / 16,777,216** | 32.48 s |

The zero-difference original-input output has Fortran-order voxel SHA-256 `1ac2c3c86d90080b7f7c7b90dad491f8e97eba2634d97dc932993088c7de2bed`. The experimental preclassified-aseg output has voxel SHA-256 `b8739d9363486baf9b85ecd274a692aecc1b3f6a790ef17960758e386309ac74`. Ordinary Euclidean CC distance left 111 differences; the source-guided fast-marching initialization and float32 update eliminated them. Both comparisons still supply a native cutting mask.

The official `mri_fill` reference took 41.25 s on control-host with diagnostic intermediate writing enabled and 40.04 s without diagnostics. Both native reruns matched the existing `filled.mgz` at all 16,777,216 voxels. The 4.69 s figure is only the segmentation fill after CC classification and the cutting mask are supplied; the 32.48 s figure includes Python CC fast marching but still takes the native cutting mask as input. A second Python run took 32.86 s. These are not end-to-end recon-all timings or an equivalent speed comparison.

To rerun the exact-core gate from a machine with the NFS paths and project Python dependencies, set `PYTHONPATH` to the repository `src` directory and run:

```bash
BASE=/data/workspaces/freesurfer_synth/work
SUBJ=$BASE/reconall_main_20260924/single_subjects/fs_sub01/mri
REF=$BASE/reconall_python_gpu_20260925/fill_ref
python validation/recon_all/python_gpu_port/experimental/check_fill_aseg.py \
  "$SUBJ/wm.mgz" "$SUBJ/aseg.presurf.mgz" "$SUBJ/filled.mgz" \
  --cc-cut-mask "$REF/cc.mgz"
```

This initial core check supplied an official cutting mask. The full-stage check below removes that dependency.

## Full Talairach and aseg-guided stage

The LTA type-0 VOX_TO_VOX matrix drives the same trilinear WM and nearest-neighbor aseg resampling as source `MRItoTalairachEx`. The Python seed search counts an 11×11 sagittal neighborhood, using the source's first-maximum scan order. It found `(124, 109, 92)` with score 103. Four-connected region growing, inferior extension of the cut, neighbor-based non-midline erasure, and reverse trilinear resampling produce the input-space mask. The source eraser accepts a non-WM center when the left and right neighbors are WM with different labels; requiring a WM center dropped two Tal voxels and seven final mask voxels, so the production code preserves that source behavior. The output writer embeds `TAG_OLD_COLORTABLE` version 2 from the 281-byte ASCII LUT.

Additional fixed input SHA-256 values: `transforms/talairach.lta` is `d5996cbdb5b1fc547a5ca71d3c8bc797f1cc259df177e8deba871c3125b82b1d`; `SubCorticalMassLUT.txt` is `65957c41244153d6614aeaf08c16ee956c09e2c8b5e198aa64c81946f75e93b6`. A diagnostic official rerun captured the transient Talairach-space cut immediately after `mri_erase_nonmidline_voxels`: `fill_ref/capture/cc_tal_captured.mgz`, SHA-256 `cfebc0480dc3539fee09312b3f511655e67d48fa7bc8f844fde329f121ffda80`. That capture has 331 nonzero voxels. The final official input-space `cc.mgz` has 589.

| Comparison with official fs_sub01 | Mismatched voxels |
| --- | ---: |
| WM Talairach binary sagittal slice | 0 / 65,536 |
| Four-connected CC region | 0 / 65,536 |
| Extended sagittal cut | 0 / 65,536 |
| Talairach-space CC cut after aseg filtering | 0 / 16,777,216 |
| Reverse-mapped source-space CC cut | 0 / 16,777,216 |
| Complete `filled.mgz` versus original official `filled.mgz` | **0 / 16,777,216** |

The complete Python output voxel SHA-256 in FreeSurfer Fortran order is `1ac2c3c86d90080b7f7c7b90dad491f8e97eba2634d97dc932993088c7de2bed`. MGH 284-byte header, 20-byte scan parameters, affine, data type, all non-command footer tags (including the 255-byte embedded color-table tag), and the 207-byte `ponscc.cut.log` are exact against the original official reference. Both outputs have eight command-history tags; the new final tag names the Python stage instead of the native executable. The compressed Python MGZ file SHA-256 is `3efcf1f06cb6b6cc9565fed52d858f32a960f7fa6fb1bb91b1e681cae66802c2`; the native compressed file has a different hash because the final command provenance and compression bytes differ. This is a voxel and non-command metadata match, not a compressed-file byte match. `check_fill_full.py` prints each gate.

On control-host, full Python CLI runs took **37.30 s** and **35.67 s** including input/output and imports with a warm Numba cache. The earlier isolated native no-diagnostics run took **40.04 s** on the same host; its output was voxel-identical to the original official output. These are individual timings, not a robust speed benchmark. All computations in this stage run on CPU. The runtime requirements are Python, NumPy, SciPy, Numba, nibabel, the input `wm.mgz`, `aseg.presurf.mgz`, type-0 `talairach.lta`, and the small color table. No FreeSurfer binary or runtime bundle is needed for the Python command.

To rerun the complete stage and validation with the fixed shared-NFS reference:

```bash
BASE=/data/workspaces/freesurfer_synth/work
SUBJ=$BASE/reconall_main_20260924/single_subjects/fs_sub01/mri
REF=$BASE/reconall_python_gpu_20260925/fill_ref
LUT=$BASE/reconall_main_20260924/bundle/SubCorticalMassLUT.txt
PYTHONPATH=src python -m fnit.recon_all.fill_cutting_plane_python \
  "$SUBJ/wm.mgz" "$SUBJ/aseg.presurf.mgz" "$SUBJ/transforms/talairach.lta" \
  "$LUT" "$REF/fill_full_python.mgz" --cut-log "$REF/ponscc_python.log"
PYTHONPATH=src python validation/recon_all/python_gpu_port/experimental/check_fill_full.py \
  "$REF/fill_full_python.mgz" "$SUBJ/filled.mgz" "$LUT" \
  --cut-log "$REF/ponscc_python.log" --reference-cut-log "$REF/ponscc.cut.log"
```

The Python API is `fill_mgz(wm_file, aseg_file, lta_file, colortable_file, output_file, cut_log_file=None)`. The tested path is the aseg-guided recon-all branch with a type-0 LTA and matching WM/aseg geometry. Other `mri_fill` option branches, other subjects, and cold-cache timing remain unvalidated. The native package was used only to generate comparison references.

An adjacent-stage replay supplied the independently generated Python `mri_em_register` LTA instead of the native LTA, while keeping the official `wm.mgz` and `aseg.presurf.mgz` fixed. The Python fill still matched all 16,777,216 native `filled.mgz` voxels, its MGH header and non-command tags, and the 207-byte cutting-plane log. The generated LTA differs from native by one float32 ULP in one matrix element, so this checks the registration-to-fill interface without rerunning the full subject. See [`em_fill_chain_report.json`](../em_fill_chain_report.json).
