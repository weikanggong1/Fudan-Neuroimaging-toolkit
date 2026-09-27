# Seven-target Conda rebuild: binary and numerical provenance

On 2026-09-27 the builder gained `mri_segment` as a seventh target. The original six binaries had been built on gpucw1 and used by the completed v2 end-to-end run. The revised builder was then run first on headcw and again on gpucw1 against the same fixed FreeSurfer commit `d932c45b7941662ea380a05efef580568b98d41a` and Conda prefix. Both seven-target builds exited successfully. **Neither build reproduced any of the six earlier binary SHA-256 values.** Their numerical relationship to v2 therefore requires separate checks; the seven-target builds were not used for the completed v2 end-to-end run.

## Why binary identity changed

The original configure log `logs/fs_cpp_configure9.log` identified gpucw1 as `HOST_OS=CentOS7`; the headcw seven-target log identified `HOST_OS=Rocky8`. Headcw's CMake configuration switched `QT_SYSLIBS` from `OFF` to `ON` and added `-Wno-cpp -Wno-restrict -Wno-format-overflow -Wno-bool-compare` to the C++ flags. The rebuilt `mris_sphere.cpp` had unchanged `-D` macros, architecture options, and `-O3`; the four warning controls were its only added compile tokens. Its actual old and new link commands did not include `-static-libgfortran`, despite a change in CMake's generic linker diagnostic. The headcw build ran 206 steps: 198 C/C++ objects, one `libutils.a`, and seven executable links.

The subsequent **gpucw1 seven-target build** restored the CentOS7 CMake branch, `QT_SYSLIBS=OFF`, and the original C++ flags, but again ran 206 build steps and recovered **0/6** old hashes. Its `mris_sphere` compile command had the same macros and compiler/optimization flags as the archived old command; the link-map output path was the only differing link token. A separate source of binary variation is explicit in FreeSurfer `utils/version.cpp:269`: `getAllInfo` embeds `__DATE__ " " __TIME__` when `libutils.a` is compiled. The retained binaries contain different build-time string prefixes (`...10:5...` original, `...13:4...` headcw, `...13:5...` gpucw1). Thus the headcw OS branch caused a rebuild and changed compilation settings, **but OS difference alone does not explain the persistent SHA changes on gpucw1**; rebuild time is at least one additional byte-level cause. These observations do not by themselves prove a numerical difference.

| Binary | v2 six-target SHA-256 | headcw seven-target SHA-256 | gpucw1 seven-target SHA-256 |
| --- | --- | --- | --- |
| `mri_em_register` | `e88735766b84d2f2d5176db56aa1f85477f81f355bad4101f24e7df678966713` | `6b21e81101f9815589cc5c5ef0c38d541a88c5f2ba70e2e69e993bee15ac6fd9` | `3139920809ebbc62160192d3f79635d174a950f8f6e7f7f5568472d044c665f3` |
| `mris_fix_topology` | `efeeb36fa125daee72f2df893a2bf9f426607a9218f7320cd5c56872a0eb44a4` | `c15e002601ac99d89ca3bd952828f9660eef21fec36ffe3ced3d2af5db7418ab` | `562856e5a60fdd0b7471a72de01d6647e48dedc619ae8a021b5be30c68d1b71c` |
| `mris_inflate` | `5625ac51327851860859265729be55246b1b31ddba8b001f8f68739b1066f2a2` | `270a93f8a56268571ac21db8a01343457ba12aeb6ab03eb2a8b6b423a79876b0` | `bfe65e1a6b1344e3738192b12781f77085c7ae60fbfeb2062d7e274c287a0588` |
| `mris_place_surface` | `feaa72b02121c5d29eac94836417525a8674869d0513659506e0254a09c0a670` | `5ea443bb624ba4e797a7a4c06c995608cb61e194b81440533c66a7ebe084c41b` | `9a42f5d7b70a066daf12e67fb6a0924048b4778186ea772e8976722b226b35d5` |
| `mris_register` | `412c9653e0c70ef3e99d06bf5fd909a94426763cee90434310942c6bee0852e5` | `5b4adca8c3f94034a3b5d3e87feb235de5343502729f8ab0ec7e97587f90414b` | `dca843e698ec65d6246f5c33c49ad647307a86aa44949a2aad020630df53bed1` |
| `mris_sphere` | `ba319a1fe45b4a5aea7c3438d9c2fe0bb062b6f731caf71709baf998b9b78a99` | `6149c2069a3211e4258a5d1917e4aa6d96f786d909f2afa441d9e5e50077629a` | `d31406ee5e2bdb41276f23414cc632371b64e206d646c0d907cfcda0f2cd8b1a` |
| `mri_segment` | not built | `cf0ba5982db19aefbf2bd25aa0fc9f314b6275850beab71269b1d4b362dcd04a` | `f3c46df3f178e932d42f52c61118696f5893e8f7def7a53eb2cbb5b7d222e5f3` |

The two seven-target hash manifests are retained as [`bin_seven.sha256`](bin_seven.sha256) and [`bin_seven_gpucw1.sha256`](bin_seven_gpucw1.sha256). The gpucw1 build wall time was 149.45 s. Source validation recorded archive-tree SHA-256 `df2ace4b904dc722090782895c251ceac65b8c52f192c2abcf7ce3daabc83585`.

## Frozen-input `mris_sphere` comparison

Each seven-target `mris_sphere` was copied unchanged to an isolated gpucw1 trial directory. Each trial copied the saved official subject, removed its `lh.sphere`, then matched the **111-file input SHA-256 manifest byte for byte** (manifest SHA-256 `6b209e92dd1565543082418d5c08450ae6cd3c9cf773879f1b2ce52c9b01df2f`). Both used `-threads 4 -seed 1234` on the same `lh.inflated`. The exact commands and file hashes are in the [headcw JSON](sphere_sevenbuild_same_input.json) and [gpucw1 JSON](sphere_sevenbuild_gpucw1_same_input.json).

| Output | Wall time | Versus v2 Conda coordinates/faces | Mean vertex distance to official | Maximum distance to official |
| --- | ---: | --- | ---: | ---: |
| v2 Conda six-target | 338.37 s | reference | 2.434603 mm | 8.203531 mm |
| headcw seven-target | 307.00 s | **exact / exact** | 2.434603 mm | 8.203531 mm |
| gpucw1 seven-target | 317.90 s | **exact / exact** | 2.434603 mm | 8.203531 mm |
| Official FreeSurfer 8.2 | 376.62 s | differs | 0 mm | 0 mm |

For both seven-target outputs, all **117,777 ordered vertices** have exactly the same XYZ values as v2 Conda (zero differing coordinate scalars; maximum absolute difference 0 mm), and all **235,550 ordered faces** match. Their surface-file SHA-256 values differ because the files contain non-geometric data: the first bytes identify creation times `12:22:29` (v2), `13:51:42` (headcw build), and `14:02:54` (gpucw1 build) on the same day; their coordinate and face arrays match. The initial scale remains `0.332` for all Conda builds versus `0.333` official. These separate shared-node timings are observations, not an equivalent-reconstruction speedup.

This frozen sphere case establishes numerical continuity of `mris_sphere` across the three Conda binaries despite distinct hashes. It does **not** establish same-input parity for the other rebuilt programs or full end-to-end equivalence for the seven-target build. The completed v2 end-to-end/strict output results apply to the original six-target binary set only.
