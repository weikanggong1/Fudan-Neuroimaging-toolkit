# Seven-target Conda rebuild: binary and numerical provenance

On 2026-09-27 the build script was extended to compile `mri_segment` as a seventh target. The revised builder was run on **headcw (Rocky8 CMake branch)** against the same fixed FreeSurfer source commit `d932c45b7941662ea380a05efef580568b98d41a`, in the same Conda prefix and existing `fs_cpp` output tree. The v2 end-to-end run had used the earlier six binaries built on **gpucw1 (CentOS7 CMake branch)**. The new build completed, but every prior binary SHA-256 changed. Those two binary sets must be identified separately in all benchmark records.

## Why the hashes changed

The archived v2 configure log `logs/fs_cpp_configure9.log` says `HOST_OS=CentOS7`, whereas `build_seven_20260927.log` says `HOST_OS=Rocky8`. The latter reconfigured Ninja and ran 206 build steps: 198 C/C++ objects, one static `libutils.a`, and seven executable links. Its CMake diagnostic changed `QT_SYSLIBS` from `OFF` to `ON` and added `-Wno-cpp -Wno-restrict -Wno-format-overflow -Wno-bool-compare` to `CMAKE_CXX_FLAGS`. The generic linker diagnostic also omitted `-static-libgfortran`; inspection of the actual old and new `mris_sphere` link commands found that neither target link line contained this option. The actual `mris_sphere.cpp` compile commands had identical `-D` macros, architecture options, and `-O3`; the four warning controls were the only added compile tokens. The same copied source and Conda toolchain were used. These facts explain why Ninja rebuilt and why a different binary identity requires validation; they do not establish a numerical change.

| Binary | v2 six-target SHA-256 | headcw seven-target SHA-256 |
| --- | --- | --- |
| `mri_em_register` | `e88735766b84d2f2d5176db56aa1f85477f81f355bad4101f24e7df678966713` | `6b21e81101f9815589cc5c5ef0c38d541a88c5f2ba70e2e69e993bee15ac6fd9` |
| `mris_fix_topology` | `efeeb36fa125daee72f2df893a2bf9f426607a9218f7320cd5c56872a0eb44a4` | `c15e002601ac99d89ca3bd952828f9660eef21fec36ffe3ced3d2af5db7418ab` |
| `mris_inflate` | `5625ac51327851860859265729be55246b1b31ddba8b001f8f68739b1066f2a2` | `270a93f8a56268571ac21db8a01343457ba12aeb6ab03eb2a8b6b423a79876b0` |
| `mris_place_surface` | `feaa72b02121c5d29eac94836417525a8674869d0513659506e0254a09c0a670` | `5ea443bb624ba4e797a7a4c06c995608cb61e194b81440533c66a7ebe084c41b` |
| `mris_register` | `412c9653e0c70ef3e99d06bf5fd909a94426763cee90434310942c6bee0852e5` | `5b4adca8c3f94034a3b5d3e87feb235de5343502729f8ab0ec7e97587f90414b` |
| `mris_sphere` | `ba319a1fe45b4a5aea7c3438d9c2fe0bb062b6f731caf71709baf998b9b78a99` | `6149c2069a3211e4258a5d1917e4aa6d96f786d909f2afa441d9e5e50077629a` |

The seventh `mri_segment` SHA-256 in the headcw build is `cf0ba5982db19aefbf2bd25aa0fc9f314b6275850beab71269b1d4b362dcd04a`. The revised builder's source validation records archive-tree SHA-256 `df2ace4b904dc722090782895c251ceac65b8c52f192c2abcf7ce3daabc83585`.

## Frozen-input `mris_sphere` check

Before the subsequent gpucw1 rebuild, the headcw `mris_sphere` binary was copied unchanged into an isolated gpucw1 trial directory. The original official paired subject was copied, its prior `lh.sphere` removed, and its 111-file input manifest matched byte for byte (SHA-256 `6b209e92dd1565543082418d5c08450ae6cd3c9cf773879f1b2ce52c9b01df2f`). The command used `-threads 4 -seed 1234` and the same frozen `lh.inflated` input; the [JSON result](sphere_sevenbuild_same_input.json) records the exact path and command.

The headcw seven-target output and archived v2 Conda output have **exactly identical ordered coordinates for all 117,777 vertices and identical 235,550 ordered faces** (zero differing coordinate scalars; max absolute difference 0 mm). Their file SHA-256 values differ (`2d96d2293f8c7d015745d68f16c50bc39dc29c687f5f26870a03332c5e3d15ba` versus `4bce6fb82f8bbd0bd06a8b904e3a480044f048b230f5276e9699db9d78c1288f`), consistent with non-geometric surface header data such as creation time. Both have the same 0.332 first scale and the same distance to the official output: mean 2.434603 mm, maximum 8.203531 mm. The new trial wall time was 307.00 s versus archived v2 Conda 338.37 s and official 376.62 s; these separate shared-node runs are not a paired speed claim.

This one frozen sphere case shows numerical continuity despite binary hash changes. It does not extend v2 end-to-end or all-stage numerical acceptance to all six rebuilt binaries. A gpucw1 rebuild using the revised seven-target builder was started separately to restore the target-node CMake branch; its final hashes and acceptance belong in a follow-up record.
