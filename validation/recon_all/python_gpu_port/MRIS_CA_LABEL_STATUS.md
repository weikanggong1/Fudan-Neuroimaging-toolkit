# Native-free `mris_ca_label` on the frozen `fs_sub01` surfaces

The six FreeSurfer 8.2 cortical annotation calls for `fs_sub01` now have a
Python/PyTorch implementation. It reads the existing `smoothwm`, `sphere.reg`,
`aseg.presurf.mgz`, and `cortex.label` inputs plus six external GCS atlases and
`ic4.tri`/`ic7.tri`. It calls no FreeSurfer executable or library and writes
the FreeSurfer v2 color-table `.annot` format. The source reference is
FreeSurfer commit `d932c45b7941662ea380a05efef580568b98d41a`, especially
`mris_ca_label.cpp`, `utils/gcsa.cpp`, `utils/mrisurf_metricProperties.cpp`,
`utils/mrisurf_topology.cpp`, and `utils/colortab.cpp`. Its seeded permutation
uses [VNL's `vnl_random` sequence](https://github.com/vxl/vxl/blob/master/core/vnl/vnl_random.cxx)
as called by FreeSurfer `OpenRan1`.

## Exact-byte validation

All six Python outputs matched the saved official `.annot` **byte for byte**.
Six fresh native runs with the same official arguments and no diagnostic
snapshot option also matched those official files byte for byte. The
[evidence JSON](gcsa_label_six_atlas_control-host.json) records SHA-256 for every
surface, cortex label, atlas, template mesh, aseg volume, and output; it also
records each Python substep and native wall time. The common aseg SHA-256 is
`263653b66a9aacba4d0d704781c86a9f99374e37d7e3f28d1674ce2`.

| Output | Vertices | Python CPU s | Native s | Output SHA-256 prefix |
| --- | ---: | ---: | ---: | --- |
| `lh.aparc.annot` | 106,622 | 25.32 | 5.37 | `9b4b6dce4703` |
| `rh.aparc.annot` | 105,541 | 24.69 | 6.39 | `aafe54c8b3ae` |
| `lh.aparc.a2009s.annot` | 106,622 | 31.62 | 7.75 | `decd0719831b` |
| `rh.aparc.a2009s.annot` | 105,541 | 33.71 | 8.48 | `1dba46c530c3` |
| `lh.aparc.DKTatlas.annot` | 106,622 | 23.80 | 6.49 | `fde919d2f8c4` |
| `rh.aparc.DKTatlas.annot` | 105,541 | 21.49 | 5.17 | `055cb307321d` |

The Python timer starts inside `label_surface`, after interpreter imports. The
native timer includes process startup. Runs were sequential on `control-host`, so
these are observed single-run times rather than a repeated speed benchmark.
The current Python implementation is slower than native for this stage; exact
annotation output was the acceptance target.

The first classifier pass and first aseg relabel of LH/RH DK matched the fresh
native `_000` snapshots at **106,622/106,622** and **105,541/105,541** vertex
IDs. Each of the ten seeded Gibbs snapshots matched all vertex IDs and native
changed/examined counts. The second aseg pass changed 221/179 labels, exactly
the native counts. Starting with the Gibbs endpoint, Python island relabeling
matched native `_post` annotations at all vertices. The LH DK island
intermediate counts differed (Python 133 then 3 versus native 135 then 1), but
the completed island output and the final `.annot` were exact. From native
`_post` snapshots, Python mode filtering plus cortex correction also matched
both final DK annotations at all vertices.

The four focused binary-parser, GCS neighbor, medial-wall fallback, and
annotation writer tests passed on `control-host` (`4 passed in 1.65s`). A separate
native `_000` annotation was serialized by Python with identical SHA-256
`81a7094785e46f1317c368f7efe080eee524702c32c24ceecef203e04eed9227`.

## Standalone use

Install the repository Python dependencies and fetch the six GCS files plus
`lib/bem/ic4.tri` and `lib/bem/ic7.tri` through the existing external asset
setup. For one hemisphere and one atlas:

```bash
python -m fnit.recon_all.gcsa_label_python \
  --subject /path/to/subjects/fs_sub01 \
  --hemi lh \
  --atlas /path/to/assets/average/lh.DKaparc.atlas.acfb40.noaparc.i12.2016-08-02.gcs \
  --ico4 /path/to/assets/lib/bem/ic4.tri \
  --ico7 /path/to/assets/lib/bem/ic7.tri \
  --output /path/to/lh.aparc.annot \
  --device cpu
```

Python callers can use
`fnit.recon_all.gcsa_label_python.label_surface(subject, hemi,
atlas_file, ico4_file, ico7_file, output_file, device="cpu")`. Select
`device="cuda:0"` to place the curvature and principal-direction calculations
on GPU. Atlas parsing, Gibbs reclassification, island relabeling, mode
filtering, and annotation writing currently run on CPU. All exact-byte results
above used `device="cpu"`; CUDA parity has not been measured. Neither this
module nor the six-case result is yet wired into the main recon-all runner.
The frozen upstream surfaces, aseg, and cortex labels came from official
recon-all; this validation covers the annotation stage, not a full T1-to-stats
native-free reconstruction.

For the fixed commands, atlas names are `DKaparc`, `CDaparc`, and `DKTaparc`
with their exact `2016-08-02.gcs` filenames in the evidence JSON. The v2
color table is embedded in each output file. The GCS data are 18.6–22.0 MB
each; `ic4.tri` is 0.21 MB and `ic7.tri` is 13.37 MB. They are data assets,
not FreeSurfer binaries.
