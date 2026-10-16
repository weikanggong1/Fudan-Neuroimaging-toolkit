# Initial surface smoothing

The fixed `mris_smooth -nw -seed 1234` call uses ten rounds of
`MRISaverageVertexPositions`. `smooth_surface_python.py` rebuilds the
source-ordered first-ring lists from triangle faces, adds each vertex's own
float32 coordinate, and averages ten times. It preserves the input surface's
volume geometry and coordinate tags. The CPU path is validated; an optional
PyTorch `--device cuda:N` averaging path is present but is not yet accepted
without H100 output comparison.

| Hemisphere | Vertices | Triangles | Native median | Python CPU median |
| --- | ---: | ---: | ---: | ---: |
| Left | 102,764 | 205,560 | 1.480 s | 3.247 s |
| Right | 101,454 | 202,936 | 1.864 s | 3.568 s |

The [three-trial paired report](smooth_surface_cpu_report.json) uses freshly
replayed `orig.nofix` triangles as identical input on `control-host`. In every
trial, the complete ordered vertex and face bytes match native exactly.
Native process startup is included, while Python imports are excluded.
FreeSurfer `mris_info` reads both Python outputs. The differing footer contains
native command-history text; the Python writer retains the source geometry
metadata. The [focused test](test_smooth_surface_python.py) checks neighbor
order and one averaging step.

The continuous Python
`mri_pretess → mri_tessellate → mris_extract_main_component → mris_smooth`
replay produced bilateral `smoothwm.nofix` coordinates and faces identical
to the completed subject's saved official outputs. The longer bilateral
six-stage replays ([LH](six_stage_surface_chain_lh_report.json),
[RH](six_stage_surface_chain_rh_report.json)) start from frozen `filled.mgz`
and `norm.mgz` and feed each Python output directly to the next stage.
Their `orig.nofix`, `smoothwm.nofix`, `inflated.nofix`, and
`qsphere.nofix` match every official ordered vertex, face, and
volume-geometry byte. Six Python CPU CLI calls took 186.99 s LH and
202.27 s RH in separate runs; these are not paired native speed
comparisons. Upstream filling and later spherical topology repair remain
outside this chain.

## Stage time context

| Stage | LH Python chain CLI | LH native isolated | RH Python chain CLI | RH native isolated |
| --- | ---: | ---: | ---: | ---: |
| `mri_pretess` | 3.157 s | 0.999 s | 2.866 s | 0.745 s |
| `mri_tessellate` | 2.075 s | 0.739 s | 2.084 s | 0.806 s |
| `mris_extract_main_component` | 0.380 s | 0.686 s | 0.432 s | 0.413 s |
| `mris_smooth` | 4.516 s | 1.480 s | 3.969 s | 1.864 s |
| `mris_inflate` | 41.448 s | 6.986 s | 44.206 s | — |
| `mris_sphere -q` | 135.411 s | 41.866 s | 148.712 s | — |

Each Python number is one fresh-process call in the bilateral connected
replays. Native numbers come from separate same-host isolated comparisons
documented in [pretess](PRETESS.md), [tessellation](TESSELLATE.md),
[extraction](EXTRACT_MAIN_COMPONENT.md), the smoothing paired report above,
[inflation](INFLATE_STATUS.md), and [quick sphere](SPHERE_QUICK_STATUS.md).
The RH inflation and quick-sphere calls have no clean same-input native
speed pair. These rows are context, not a paired six-stage speed ratio.
