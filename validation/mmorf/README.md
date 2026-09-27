# MMORF validation

`report.public.json` contains the de-identified one-subject comparison used by
the [MMORF function page](../../docs/mmorf/README.md). FSL MMORF 0.3.2 and
FNIT received the same brain-extracted T1w, FSL six-frame tensor, templates,
and FLIRT initial matrices. The official warp was also applied through the
FNIT sampler so the nine-map table isolates warp-estimation differences.

The output geometry and displacement units agree. The numerical fields do not:
FNIT currently uses a different optimizer, lattice interpolation, regularizer,
cost, and tensor-reorientation approximation. The report therefore records
`numerically_equivalent=false`.

The source images and subject identifier are not distributed. The repository
contains only aggregate metrics and a synthetic illustration.
