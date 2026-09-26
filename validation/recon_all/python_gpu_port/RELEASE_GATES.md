# Native-free recon-all acceptance gates

This branch contains independently validated translations. The published
`fnit-recon-all` entry point still launches a scoped FreeSurfer runtime. It
must remain described as a hybrid until the gates below pass on the same T1.

1. **Connected run:** start with the original T1 and an empty subject folder;
   generate all fixed-profile outputs using installed `fnit`, external weights,
   and verified templates. Capture the exact command, versions, input/asset
   hashes, stage timings, and process tree. The process tree must show no
   FreeSurfer executable or script call.
2. **Volumes:** compare every reconstructed volume to a fresh paired FreeSurfer 8.2
   run, including voxel values, affine, and MGH header. Report the mismatch
   count per volume; explain byte-level metadata differences separately.
3. **Surfaces:** compare ordered vertices, ordered faces, volume geometry,
   topology, and hemisphere labels for `orig`, `white`, `pial`, `inflated`,
   `sphere`, and `sphere.reg`. Report every vertex error and the first
   difference. A copied-source diagnostic is supporting evidence, not a
   substitute for the installed reference output.
4. **Vertex metrics:** compare all vertices of thickness, white/pial/mid
   area, vertex volume, mean/Gaussian curvature and other emitted maps with
   the existing per-map numerical rules. Record maximum error, error
   distribution, and outlier count for each hemisphere. Zero outliers is
   required; a summary mean cannot replace the vertex comparison.
5. **Annotations and statistics:** compare ordered annotation IDs and every
   ROI row and global measure for aparc, a2009s, DKTatlas, BA/exvivo,
   aseg, and wmparc. Apply the existing printed-precision and numerical
   rules to each column, and record any formatting differences.
6. **Timing:** after the connected run passes, time each candidate and native
   stage on the same host and same input, including file I/O. Record warm and
   cold starts separately, GPU initialization, thread counts, device, and
   shared-host load. Do not combine isolated CPU measurements with H100
   measurements into a full-flow speed ratio.

A connected Python T1-to-N4 run matched a fresh same-host FreeSurfer N4
voxel for voxel, but differed from the archived unmodified official full run
at 34/16,777,216 voxels; see [the report](CONNECTED_N4_20260926.md).
That archival discrepancy remains open for the strict volume gate.
The connected SynthSeg H100 comparison matched all 16,777,216 hard labels
only with SynthSeg cuDNN TF32 disabled. Its 33 soft-volume columns still differ
by up to 0.04 mm³ after a same-input threshold correction and source-style
float32 CSV rendering; six columns exceed the existing 0.005 mm³ stats
tolerance. The rendering result is a deterministic replay of saved values,
not another GPU inference; see
[the report](CONNECTED_SYNTHSEG_GPU_20260926.md). The CSV gate remains open.
The following Python T1 normalization matched a fresh native command
on the same Python-generated `nu.mgz` input, while differing from the archived
full subject at 112 voxels ([report](CONNECTED_T1_NORMALIZE_20260926.md)).

A connected experimental native-free core now runs from T1 to bilateral morphometry,
but passes only 6/138 strict output checks on the same T1
([full report](NATIVE_FREE_CONNECTED_20260927.md)). The open critical path
still includes full topology repair, independent white placement, and exact
connected output parity. Both pial optimizers
now independently select their full 41-step schedules and match final surfaces
and vertex metrics when started from frozen official white surfaces. The
standalone conventional-sphere stage now matches both final native meshes and volume geometry on the frozen subject;
its optimization still runs on CPU/Numba and is not yet wired into a connected
T1-to-metrics pipeline ([stage report](SPHERE_STANDARD_STATUS.md)). The
continuous source-scheduled smoothwm stage now matches all LH/RH updates and
generates both final registered spheres from exact sulc seeds. The sulc stage
now independently generates those exact bilateral seeds from `sphere`. A fresh
one-call bilateral CPU/Numba registration from `sphere` matches new unmodified
FreeSurfer 8.2 final registered surfaces at every ordered vertex, face and
volume geometry field; its input/seed hashes and all selected steps also agree.
Exact ordered-topology caching reduced the observed one-call times to
372.85/347.36 s LH/RH, still slower than fresh native controls on the shared
host. The final LH overlap repair matches on both CPU and H100 CUDA. See
[the stage report](MRIS_REGISTER_STATUS.md). The stage inventory in
[README.md](README.md) states which individual boundaries already pass.

The [subject comparator](compare_complete_subject.py) checks 138 outputs in
the fixed profile: all 39 archived MRI image files, two surface MGH maps,
18 ordered meshes, 44 per-vertex morph maps including additional curvature
maps, 12 annotations and 23 statistics files. It compares every voxel and
vertex, affine and image header, ordered faces, surface volume geometry,
annotation IDs and numerical statistics fields. The archived subject passed
138/138 self comparisons ([report](complete_subject_self_check_expanded_20260926.json)).
A 0.02 mm change at LH thickness vertex 1,234 failed only that map, 137/138
([report](complete_subject_thickness_negative_expanded_20260926.json));
a 0.02 mm² change at LH `area.pial` vertex 1,234 likewise failed only that map,
137/138 ([report](complete_subject_area_negative_expanded_20260926.json)).
Both controls used [linked subject trees](experimental/create_morph_negative_control.py)
and reported vertex 1,234 as the sole outlier. These checks validate the
comparator, not a new reconstruction. A fresh paired FreeSurfer run and a complete, numerically matching
native-free candidate are still required for acceptance.

An [existing unmodified FreeSurfer subject comparison](TRUE_OFFICIAL_BASELINE_20260926.md)
shows that the archived hybrid subject passes 110/138 checks against the true
official run on the same T1. Ordered mesh coordinates/faces, all 46 surface
scalar files and annotations agree; SynthSeg soft volumes, SynthMorph warps,
several derived statistics and label storage types differ. The float32 label
storage issue has been corrected in the current independent SynthSeg source,
and the first connected approximate native-free core has now been compared
against that reference (6/138 strict outputs pass). Exact complete parity
remains open; see [the connected report](NATIVE_FREE_CONNECTED_20260927.md).
