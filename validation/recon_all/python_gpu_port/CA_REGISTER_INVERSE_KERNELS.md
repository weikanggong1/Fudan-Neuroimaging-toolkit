# Fixed `mri_ca_register -invert-and-save` parity

## Scope and result

This isolated Python implementation reproduces the complete fixed FreeSurfer
8.2.0 `fs_sub01` inverse-warp call in `recon-all.log` line 573:

```bash
mri_ca_register -invert-and-save warp.to.mni152.1.0mm.1.0mm.nii.gz warp.to.mni152.1.0mm.1.0mm.inv.nii.gz
```

The frozen input SHA-256 is
`b1bf39f633fa6b3ca252472c569d7e7007a4af8e76e46df5d601abdcdcbf9005`.
It has a 193 × 229 × 193 atlas grid, 256³ source grid, three float32 RAS
displacement components, and a FreeSurfer geometry extension. The original
output and two fresh native runs have identical compressed SHA-256
`1b5b6402c1228f37ad7680b062ab3a17bf0e78dafd0049d769b1698466443498`.
The final isolated Python writer produced that same compressed SHA-256.

The implementation is **CPU** NumPy, Numba, and SciPy. It calls no FreeSurfer
executable. It is intentionally limited to this fixed 256³ geometry and the
observed quaternion branch. It has not been inserted into the main recon-all
runner or tested on another T1 input; the whole recon-all pipeline is not
covered by this result.

## Native path and numerical checkpoints

Source commit: `d932c45b7941662ea380a05efef580568b98d41a`.
The active branch is `mri_ca_register.cpp:2216–2230` (`GCAMread`, `GCAMinvert`,
`GCAMfillInverse`, `GCAMwrite`). The port follows
`utils/warpfield.cpp:412–555,678–750`, `utils/gcamorph.cpp:7501–7640`,
`utils/mri.cpp:9573–9635`, `utils/mrinorm.cpp:1795–1934,2367–2512`,
`utils/mriio.cpp:9647–9775,10229–10398`, and the FreeSurfer extension tags in
`utils/fstagsio.cpp`.

The native diagnostic capture used `mri_warp_convert --out-interp abs-crs`,
`DIAG=0x8 DIAG_VERBOSE=1 mri_ca_register`, and watchers for the raw and
post-Voronoi x fields. The Python implementation is split into
`ca_register_inverse.py` (coordinate conversion and trilinear splat),
`ca_register_inverse_fill.py` (27-neighbor Voronoi and 50 soap-bubble passes),
and `ca_register_inverse_output.py` (RAS displacement and NIfTI writing).

| Fixed-input comparison | Elements | Different |
| --- | ---: | ---: |
| Source CRS coordinates | 25,590,063 | 0 |
| Trilinear count and raw x sum | 16,777,216 each | 0 each |
| Control mask and marked x average | 16,777,216 / 2,799,105 | 0 / 0 |
| x field after Voronoi | 16,777,216 | 0 |
| Final x, y, z fields after soap-bubble | 16,777,216 each | 0 each |
| Final RAS displacement | 50,331,648 | 0 |

Each axis required 91 Voronoi shells and 50 soap-bubble iterations. The input
extension provides the two volume geometries. Matching the native float32
matrix inversion and quaternion operations was necessary for exact output.

## File-level gate

The NIfTI output has shape `(256, 256, 256, 1, 3)` with the same affine as
native. The raw 348-byte header SHA-256 is
`14419ac8dda80bc3b2e9a7956c5d02d0051fd265ca13a321aa0cd750eb0a8c54`.
The 67,109,744-byte FreeSurfer extension, including its eight-byte NIfTI tag
header and padding, has SHA-256
`1ff80fffbf78a4ee00bf78e3993dbe351ac245508ffe312d69b22afca1ad47df`.
The entire decompressed NIfTI has 268,436,688 bytes, zero differing bytes,
and SHA-256 `9cc9a86ada9cc0b7873136d123b7c8756ee5081cbe907b284164ac19de4661b5`.

The first full Python run used the default gzip filename and current timestamp;
its decompressed NIfTI was exact but its compressed SHA-256 differed. The final
writer uses the native gzip header and reproduces the complete compressed file
byte for byte. `ca_register_complete_file_comparison.json` records the first
full run, and `ca_register_writer_file_comparison.json` records the final
writer replay and compressed hash match.

## Timing

An isolated sequential pair on control-host used the same input and no diagnostic
checkpoint writing. `/usr/bin/time -p` measured **84.68 s native** and
**204.82 s Python** (Python/native 2.42×). Both commands used approximately
one CPU core. Their compressed NIfTI outputs passed `cmp` and had SHA-256
`1b5b6402c1228f37ad7680b062ab3a17bf0e78dafd0049d769b1698466443498`.
The Numba cache was warm from the earlier parity run. System load average
changed substantially across this sequential pair: 97.05 before native,
130.09 between commands, and 17.95 after Python, on a 192-logical-CPU host.
This is one observed pair, not a load-controlled speed estimate. It shows no
speed advantage for the present CPU port.

Python's internal 203.90 s comprised 2.38 s input/conversion/splat,
63.73/64.73/63.24 s x/y/z field generation, 1.60 s displacement conversion,
and 8.22 s gzip writing. The three fields consumed about 94% of the time.
The exact logs, load snapshots, SHA manifest, and timing JSON are in
`ca_register_pair_benchmark/`.

For context, the original recon-all log reports 128.93 s for native, while a
fresh native run with diagnostic writes took 132.28 s. A separate concurrent
diagnostic native run took 104.18 s. A full Python diagnostic run took
211.75 s including native checkpoint reads, JIT initialization, full-file
comparison, and gzip writing while an EM GDB job was active. Those durations
are not directly comparable to the sequential no-diagnostics pair.

## Reproduction

With `FS_LICENSE` exported, `run_ca_register_inverse_reference.sh` captures
native diagnostics from `WARP` into `OUTPUT_DIR`. `benchmark_ca_register_complete.py`
compares the three final fields, final displacement, and, when passed
`--inverse-out OUTPUT_NII_GZ`, the full written file to those diagnostics.
`run_ca_register_inverse_python.py WARP OUTPUT_NII_GZ --timing-json TIMING_JSON`
runs the Python stage without native checkpoint reads. On control-host,
`benchmark_ca_register_pair.sh WARP OUTPUT_DIR PYTHON_BIN SOURCE_DIR FS_LICENSE`
runs the native and Python stages sequentially and requires compressed-byte
equality.
