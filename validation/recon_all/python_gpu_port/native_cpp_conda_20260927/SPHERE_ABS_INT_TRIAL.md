# `mris_sphere` legacy `abs` compatibility trial

Status on 2026-09-27: the isolated trial completed on gpucw1 after the concurrent end-to-end run exited successfully. It corrected the first printed scale value but **failed ordered-coordinate parity** and increased the final error. The trial did not modify the production source or binary.

## First divergence and source

On byte-identical `lh.inflated` and subject inputs, official FreeSurfer 8.2 prints `scaling brain by 0.333`; the Conda build prints `0.332` before `MRISunfold`. Both use the same source commit `d932c45b7941662ea380a05efef580568b98d41a` and `-seed 1234 -threads 4`. An official repeat at the same path reproduced every output coordinate exactly ([paired report](SPHERE_REGISTRATION_SAME_INPUT.md)).

The active source in `mris_sphere/mris_sphere.cpp:299-306` is:

```cpp
max_dim = MAX(abs(mris->xhi-mris->xlo), abs(mris->yhi-mris->ylo));
max_dim = MAX(max_dim,abs(mris->zhi-mris->zlo));
if (max_dim > .75*DEFAULT_RADIUS) {
  float ratio = .75*DEFAULT_RADIUS / max_dim;
  MRISscaleBrain(mris, mris, ratio);
}
```

`DEFAULT_RADIUS` is 100 mm. The frozen input's largest decoded coordinate span is 225.91995239 mm. The official ratio is `75 / 225 = 0.333333`; the Conda ratio is `75 / 225.91995239 = 0.33197599`. The official binary's disassembly at `0x416a7d–0x416ad9` uses `cvttss2si` before integer absolute value and converts the selected integer back with `cvtsi2ss`. The Conda binary at `main+0x7d6` (`0x16b16–0x16b74`) uses `subss`, an `andps` absolute-value mask, and `comiss`, retaining the float span. This establishes a compiler-dependent first difference in the active branch; it does not prove that this is the only source of the final 2.435 mm mean vertex distance.

## Isolated patch and build

[`mris_sphere_legacy_abs_int.patch`](mris_sphere_legacy_abs_int.patch) changes only the three active bounding-box span expressions to `abs((int)(span))`. On headcw, a copy of `fs_cpp/source/mris_sphere/mris_sphere.cpp` was patched under `$root/sphere_abs_int_trial_20260927/`. The existing Conda Ninja compilation and link commands were read with `ninja -C "$root/fs_cpp/build" -t commands mris_sphere`; only the source, object, dependency, linker-map, and output-binary paths were redirected to the trial directory. The link reuses the unchanged Conda-built static libraries from the original build tree. `commands.json`, `compile.log`, `link.log`, `ldd.txt`, `build_result.json`, and the patch are retained remotely. This compiled just the one modified translation unit and linked a separate binary; compilation took 3.21 s and linking 0.96 s.

`patch --dry-run -d "$root/fs_cpp/source" -p1 < "$trial/legacy_abs_int.patch"` passed without changing the original source. The trial is a diagnostic compatibility experiment, not a replacement for the current build.

| Artifact | SHA-256 |
| --- | --- |
| Unchanged source `mris_sphere.cpp` | `a3c9540ba926ead9541ff3703fb1ad9a2cdd431ff09804c234f5328054edcb2d` |
| Trial patched `mris_sphere.cpp` | `f9e08179921a1fdf159f7314ce4650ffc0f76cf36badfa3568ce44f426fc94bb` |
| Trial binary `sphere_abs_int_trial_20260927/mris_sphere` | `1ad65ead33ef271a2ba7f44ab6e31ade2d4b806e9e925bead042c6afc9c06b21` |
| Unchanged production `fs_cpp/bin/mris_sphere` | `ba319a1fe45b4a5aea7c3438d9c2fe0bb062b6f731caf71709baf998b9b78a99` |

The trial binary's disassembly now includes `cvttss2si` at the same bounding-box calculation. Its highest required glibc symbol is `GLIBC_2.14`; `ldd` found no missing library or installed FreeSurfer path. The compiler remains Conda GCC 11.4 and the libraries remain Conda ITK/VNL and libgomp, so the patch isolates this one source-level compatibility behavior.

## Frozen-input result

The exact [trial script](run_sphere_abs_int_trial.sh) ran on gpucw1 in a new subject copy after `reconall_conda_cpp_cc_e2e_20260927/reports/e2e.status` became `exit=0`. It copied the official paired subject, removed only its `lh.sphere` output, and matched the saved **111-file full input SHA-256 manifest byte for byte** (manifest SHA-256 `6b209e92dd1565543082418d5c08450ae6cd3c9cf773879f1b2ce52c9b01df2f`). The command was:

```bash
export FS_LICENSE=/path/to/private/license.txt
bash validation/recon_all/python_gpu_port/native_cpp_conda_20260927/run_sphere_abs_int_trial.sh
```

On gpucw1 the script was copied to `$root/sphere_abs_int_trial_20260927/run_trial.sh`; its SHA-256 was `e2fca9aa9808df280d5e1c1ae93f420d046beb22f3979024dbf9c42ded69cb1e`. It invoked `/usr/bin/time -f '%e' -o "$trial/wall_seconds.txt" timeout 720 "$trial/mris_sphere" -threads 4 -seed 1234 "$surf/lh.inflated" "$surf/lh.sphere"` from the copied subject's `scripts` directory. `FS_LICENSE` pointed to the existing private license; its contents were not copied or logged. Full values are in [`sphere_abs_int_trial_result.json`](sphere_abs_int_trial_result.json).

| Version | First scale | Wall time | Mean vertex distance to official | Maximum vertex distance | Ordered faces |
| --- | ---: | ---: | ---: | ---: | --- |
| Official FreeSurfer 8.2 | 0.333 | 376.62 s | 0 mm | 0 mm | reference |
| Original Conda build | 0.332 | 338.37 s | 2.434603 mm | 8.203531 mm | exact |
| Isolated integer-`abs` trial | 0.333 | 352.02 s | **5.715858 mm** | **13.410164 mm** | exact |

Both candidate sphere files have 117,777 ordered vertices and 235,550 ordered faces. The trial differs from the official output in 353,331 coordinate scalars; 115,565 vertices are over 1 mm away. Its output SHA-256 is `67e4ee2f3f1a5ca43e617b8f5d6e90657dc4159f5b289ccd74b444a784039e8d`, versus official `f52648c164dbebb157939afb5f4dc4f2a7c117e6e6bdb951a3afa94daee98415`. The original production binary SHA-256 was still `ba319a1fe45b4a5aea7c3438d9c2fe0bb062b6f731caf71709baf998b9b78a99` when the trial ran; later independent builds must be assessed separately.

The next visible numerical divergence is already inside the first `MRISunfold` pass: `pass 1: epoch 1 of 3 starting distance error` is 20.17% official, 20.10% trial, and 20.07% original Conda. The trial then exits `MRISunfold` with 406 negative triangles versus 396 official and 388 original Conda. Thus reproducing the legacy bounding-box truncation fixes one proven compiler-dependent operation but does **not** reproduce the downstream optimization. This isolated patch is not suitable for the production build. The independent `mris_register` discrepancy also remains unresolved.
