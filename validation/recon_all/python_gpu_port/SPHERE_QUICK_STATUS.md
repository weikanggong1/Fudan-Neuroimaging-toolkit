# Fixed FreeSurfer 8.2 `mris_sphere -q` parity

## Scope and source

The frozen `fs_sub01` recon-all commands use `mris_sphere -q -p 6 -a 128
-seed 1234` on each `inflated.nofix` surface to create `qsphere.nofix`.
The source reference is FreeSurfer commit
`d932c45b7941662ea380a05efef580568b98d41a`:
`mris_sphere/mris_sphere.cpp`, `utils/mrisurf_integrate.cpp`,
`utils/mrisurf_timeStep.cpp`, and the surface geometry helpers.

`src/fnit/recon_all/sphere_python.py` translates the initial scale,
300 sphere-inflation updates, inherited vertex momentum, half-scale, and radial
projection. `sphere_quick_python.py` translates the four nonlinear-area epochs,
source-order face forces and gradient averaging, five-candidate line search,
retry momentum, reprojection, and native stopping. `quick_sphere_from_inflated`
connects both stages. This is a **NumPy/Numba CPU geometry implementation**; it does
not call a FreeSurfer program at runtime. Its CLI writes FreeSurfer triangle surfaces with preserved input volume geometry tags. It is not yet wired to a native-free recon-all runner.

## Fixed-subject numerical evidence

| Gate | LH result | RH result |
| --- | --- | --- |
| Isolated `-in 0 -N 1` quick optimizer from official projected input | Final 102,764/102,764 ordered vertices exact | Final 101,454/101,454 ordered vertices exact |
| First raw and 128-times averaged nonlinear-area gradient | 308,292/308,292 float32 components exact | 304,362/304,362 exact |
| First independent line-search step length | 2204.0380859375, exact | 2738.337158203125, exact |
| 300 inflation updates from original `inflated.nofix` | Raw positions, old momentum, half-scaled positions, and diagnostic projected positions all 102,764/102,764 exact | Full-chain final geometry exact; intermediate checkpoints not separately captured |
| Default `-N 25` quick optimizer from native optimizer-entry state | 64/64 updates, every ordered vertex exact at every checkpoint; final area and signed volume exact | Full independent chain passes; individual optimizer checkpoints not separately captured |
| Independent full `-q` geometry chain from original `inflated.nofix` | 102,764/102,764 final ordered vertices and 205,560/205,560 ordered faces exact; area 125667.02935082487 mm² and signed volume 4188235.9650802133 mm³ equal native | 101,454/101,454 vertices and 202,936/202,936 faces exact; area 125671.95697221036 mm² and volume 4188235.165947808 mm³ equal native |
| Connected Python `smoothwm → inflated → qsphere` file outputs | Final 102,764/102,764 vertices, 205,560/205,560 faces and all volume geometry fields exact | Final 101,454/101,454 vertices, 202,936/202,936 faces and all volume geometry fields exact |
| Direct Python `pretess → tessellate → extract → smooth → inflate → qsphere` | LH `orig`, `smoothwm`, `inflated`, and `qsphere` ordered geometry and volume metadata exact | RH same four outputs exact, starting from 127-filled volume |

The detailed LH 300-step inflation comparison uses a native diagnostic run
for its raw, preprojection, and postprojection surfaces. A passive GDB capture of the pinned
installed binary checks the inherited momentum. The first update's positions,
normals, sphere force, convexity force, final gradient, and momentum are all
exact. The full optimizer and final geometry are compared with a fresh native
`-W 1` run whose final ordered geometry was independently checked against the
saved official output. No native intermediate is injected into either hemisphere
`quick_sphere_from_inflated` run.

A subtle extra projection occurs at `MRISintegrate` entry after the main
projection. Omitting it leaves 4,459 last-bit LH vertex differences at the
first optimizer input and changes the final solution by a median 0.206 mm.
With that projection restored, the full LH chain uses the same 64 updates and
has zero final vertex error. The RH full chain uses 72 updates and also has
zero final vertex error. The optimizer-only replay that starts from the
captured native entry state remains a separate conditional check.

## Timing and limits

| Measurement on control-host CPU | LH | RH |
| --- | ---: | ---: |
| Python isolated `-in 0 -N 1` quick optimizer | about 71.79 s | about 71.89 s |
| Python default optimizer only, given captured native entry state | 82.44 s | Not measured |
| Python independent inflation + default optimizer | 139.00 s | 146.40 s |
| Native full `mris_sphere -q` diagnostic with snapshots | 57.49 s | 52.24 s in a separate earlier replay |

The native diagnostic includes input/output I/O and 365 snapshot writes; the
Python optimizer-only number excludes inflation and I/O. They are not a
controlled speed ratio. A later [sequential same-input LH file-level pair](inflate_qsphere_paired_lh_control-host.json)
ran the clean native `mris_sphere -q -p 6 -a 128 -seed 1234` and the Python CLI
on the **same native-generated inflated input**: **41.866 s native versus
135.866 s Python CPU**, both including startup and I/O. Their 102,764 ordered
vertices, 205,560 faces and volume geometry fields were exact. This one pair
estimates about **3.2× slower Python CPU execution**; RH was not included in
a clean paired speed trial. The accepted bilateral geometry port has no CUDA path.

Source-order neighbor averaging, normal scatter, and spring accumulation now
use Numba. The updated connected inflation and quick-sphere CLI outputs still
match **every ordered vertex, face, and volume geometry field** on both
hemispheres. The [optimized bilateral report](inflate_qsphere_numba_optimized_report.json)
records matching input/output and source hashes. Separate control-host CLI calls
of the updated quick sphere took **56.75 s LH** and **62.27 s RH**. Those calls
were not paired with a fresh native execution or the old Python source under
controlled load; the 41.866/135.866 s LH pair above remains the only clean
native/Python timing pair and describes the earlier implementation. The
updated quick sphere remains a CPU stage.

Only one T1 subject has been checked. The actual CLI generated both quick-sphere files from official `inflated.nofix` inputs and from independently generated Python `inflated` files; all ordered vertices, faces and volume geometry fields match the archived official outputs in the [file-level CLI report](sphere_q_cli_validation.json) and [connected-chain report](inflate_to_qsphere_chain_exact_report.json). The CLI report also preserves the superseded input-sensitivity experiment: the earlier Python inflation version differed by at most 0.00027 mm at its output yet produced a quick sphere with 0.161 mm median and 6.90 mm maximum LH vertex error. The corrected inflation removes that difference. The Python surface header/provenance text differs from FreeSurfer, so whole-file hashes are different. Other subjects, recon-all runner integration, and the separate *non-quick* `mris_sphere` call remain open.
Vertex-level thickness, area, curvature, and ROI statistics require later
white/pial and registration stages and cannot be inferred from q-sphere parity.

## Reproduce

Use `validation/recon_all/python_gpu_port/benchmark_sphere_full_python.py
INPUT_INFLATED NATIVE_FINAL --output REPORT.json` with `PYTHONPATH=src` to run
the independent chain. `experimental/probe_sphere_inflation_gradient.py`,
`experimental/probe_sphere_inflation_momentum.py`, and
`experimental/trace_sphere_quick_steps.py` isolate the force, carried state,
and ordered optimizer updates. The fixed control-host outputs are under
`/data/workspaces/freesurfer_synth/work/reconall_python_gpu_20260925/tessellate/sphere/quick_full_lh_a128/first_capture/`;
`full_chain_python_vs_native_corrected.json` and
`full_chain_python_vs_native_rh.json` are the LH/RH isolated full-chain reports. The connected bilateral file-output comparison is [`inflate_to_qsphere_chain_exact_report.json`](inflate_to_qsphere_chain_exact_report.json). The longer bilateral six-stage reports ([LH](six_stage_surface_chain_lh_report.json), [RH](six_stage_surface_chain_rh_report.json)) begin at frozen `filled.mgz` and `norm.mgz` and feed every Python stage output to the next.
