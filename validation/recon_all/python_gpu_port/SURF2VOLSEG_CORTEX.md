# `mri_surf2volseg --label-cortex` atlas volume replacements

The fixed `--label-cortex` call maps `aseg.mgz` cortex voxels to the bilateral
`aparc.annot` labels using `white`, `pial`, and `cortex.label`. The Python
implementation uses PyTorch surface normals and SciPy nearest-vertex queries.
When the closest vertex faces the wrong way, it searches farther vertices
until the source's normal-direction rule holds. The output is
`aparc+aseg.mgz`. The same fixed algorithm also handles `aparc.a2009s` with
offsets 11100/12100 and `aparc.DKTatlas` with offsets 1000/2000.

On the frozen `fs_sub01` inputs, selecting the Euclidean nearest vertex
without the direction check mislabeled **1,967** cortical voxels. The source
rule triggered 15,033 such checks: 1,638/6,121 for left white/pial and
1,615/5,659 for right white/pial, matching the fresh official log's
`ndotcheck = 15033`. With that rule, all **337,081 cortical voxels** and the
entire **16,777,216-voxel** output matched both a fresh and archived official
run. The Python output was also identical to the archived output in complete
decompressed MGH bytes. The fresh native run's trailing command metadata
differs from the archived run although its voxels match. Two synthetic unit
tests passed.

A separate chained replay fed the Python-generated
`aseg.presurf.hypos.mgz` into the Python ribbon-fix stage, then fed that
Python `aseg.mgz` into this aparc stage. Both resulting volumes again had
0/16,777,216 voxel mismatches against official outputs. The inherited
trailing MGH metadata prevented full-byte identity in the chain; see the
[`chain report`](surface_volume_chain_control-host_report.json).

The other two atlas volumes also matched all **16,777,216 voxels** and the
complete decompressed MGH bytes against both the archived and fresh native
outputs. Feeding the existing Python-generated `aseg.chain.mgz` into each
branch retained zero voxel differences; annotation and surface inputs were
still frozen official files.

| Same-input control-host cold CLI | Native | Python CPU |
| --- | ---: | ---: |
| `aparc+aseg.mgz` | 27.20 s | 6.14 s |
| `aparc.a2009s+aseg.mgz` | 24.43 s | 5.96 s |
| `aparc.DKTatlas+aseg.mgz` | 22.34 s | 5.25 s |

Each pair includes process startup and file I/O; none measures a full
recon-all run.

```bash
python -m fnit.recon_all.surf2volseg_cortex_python \
  /path/to/aseg.mgz /path/to/surf /path/to/label \
  /path/to/aparc+aseg.python.mgz

python -m fnit.recon_all.surf2volseg_cortex_python \
  /path/to/aseg.mgz /path/to/surf /path/to/label \
  /path/to/aparc.a2009s+aseg.python.mgz --atlas aparc.a2009s
```

The frozen cortex-label vertices all had annotation IDs. For another subject,
the native program's five-hop search for an unannotated selected vertex may
need a separate parity check. Upstream surface/annotation generation is not
yet fully implemented in Python.
