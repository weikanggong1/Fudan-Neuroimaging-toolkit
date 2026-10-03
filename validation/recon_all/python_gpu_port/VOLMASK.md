# `mris_volmask` white and cortical ribbon replacement

> This is the historical same-input validation of the former float32 rasterizer.
> The current API, geometry-boundary fix and new real-input results are documented
> in [the ribbon function documentation](../../../docs/recon_all/VOLMASK.md).
> The times and exact matches below apply to these frozen inputs and this older implementation.

`volmask_python.py` reads `aseg.presurf.mgz` for its voxel grid, the four
bilateral white/pial surfaces, and the downloaded `FreeSurferColorLUT.txt`.
A Numba rasterizer finds where each triangle
crosses voxel-grid rays, sorts crossings, and fills the enclosed voxels. The
left white/pial mask assigns 2/3, the right assigns 41/42, and the left mask
wins at overlapping voxels, matching pinned FreeSurfer 8.2
`mris_volmask/mris_volmask.cpp`. The stage writes `ribbon.mgz`,
`lh.ribbon.mgz`, and `rh.ribbon.mgz` without a FreeSurfer executable.

On fixed `fs_sub01`, all three **16,777,216-voxel** outputs matched both the
archived and a newly run official `mris_volmask` exactly. The complete
decompressed MGH contents now match byte for byte, including the header,
voxel payload, affine, and FreeSurfer color-table footer. The footer serializes
the verified external lookup table with its supplied absolute filename; the
filename is part of the MGH metadata. The measured surface rays had no odd
crossing counts or buffer overflow (maximum 18 crossings).

| Same-input headcw CLI | Time |
| --- | ---: |
| FreeSurfer 8.2 native, four threads | 72.36 s |
| Python/Numba CPU, four surfaces and three MGZ outputs, including color tables | 3.46 s |

These same-host measurements include process startup and all input/output I/O.
The Python trial used a warm Numba disk cache. The measured speedup is about
**21 times for this isolated step**; it is not an end-to-end recon-all benchmark. A nested
cube test passed independently of the fixed subject.

```bash
python -m fnit.recon_all.volmask_python \
  /path/to/aseg.presurf.mgz /path/to/surf /path/to/mri \
  /path/to/recon-all-assets/FreeSurferColorLUT.txt
```

The frozen inputs had these SHA-256 values:

| Input | SHA-256 |
| --- | --- |
| `mri/aseg.presurf.mgz` | `263653b66a9aacba4d0d704781c6e67e581c86a9f99374e37d7e3f28d1674ce2` |
| `FreeSurferColorLUT.txt` | `da55c6a47d316ee16d2609afd5c93cb8078e08e7e3afec5bb6283027d7c33e7b` |
| `surf/lh.white` | `9c88a786c4a571c07d830b83fe647461b09fdacb63ff143670771340ef50b51c` |
| `surf/lh.pial` | `f7c61a103d48497d38fb46ff0495984624eba726f384473f5634a8d7102d66d5` |
| `surf/rh.white` | `5c7fda4367dd3624de0ef265303dab49579aa603aee75f1cce2f9c30bec965b9` |
| `surf/rh.pial` | `0e0ac9a32a6e3d8087d63be2f3a68a237bef6331e1e867d33531611dd11e1eac` |

The rasterizer stops when a ray has an odd number of intersections or exceeds
its 128-crossing buffer. The exact voxel comparison covers this subject's
closed surfaces; additional subjects and surfaces with mesh defects still
need validation.
