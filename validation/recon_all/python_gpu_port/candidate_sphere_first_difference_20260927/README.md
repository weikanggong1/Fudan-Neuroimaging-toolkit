# Candidate-input conventional sphere: first numerical difference

This is a bounded diagnostic on the deidentified real `examples/data/sub-01_T1w.nii.gz` scan (SHA-256 `f20410a4efd8e6a05cd04d55730a4a5492ecf9ad1b234fe0fd4661e448270c6a`). It does not rerun a complete sphere or use an official surface as a candidate input. The candidate `lh.inflated` and `lh.smoothwm` were generated from the FNIT white/smoothwm prefix. Their byte-exact copies supplied all native and Python probes: SHA-256 `f3a2128de469dc3c8d06e2649f85f420b26c84f95aea3c7912f677b4e57afd30` and `6d1a5d30639f3678c2e825d69b816f9bc35d5411546145f6c0bcab29a5fd0b91`. The installed FreeSurfer 8.2 `mris_sphere` SHA-256 was `c34ca308a7fa03acdb3f689bf6125cf3d0198c37c3a62992a29f68e631c73612`.

The separate full official and Python runs on these same candidate inputs are audited in [the candidate sphere control](../candidate_sphere_annotation_20260927/candidate_sphere_official_control_audit.json). Its final Python-versus-official same-index mean/P99/max displacement is 1.652636/2.952841/3.557925 mm, with ordered faces exact. The diagnostic here locates the **first** same-input difference; it does not explain the entire final displacement.

## Bounded capture and provenance

The native line-search probe used `DIAG=0x10000040`, `DIAG_VERBOSE=1`, and `FREESURFER_logSSE=1` with `mris_sphere -threads 4 -seed 1234 surf/lh.inflated lh.sphere`; it stopped immediately after four `sses:` decisions, in 29.56 s. The existing [capture script](../experimental/capture_standard_sphere_second_sse.py) had SHA-256 `532b6a74cfbe5b548ff6d229d602e9eff7198b7dfec190ae6577ab3ba27215e7`. Its original 150,418-byte log SHA-256 is `24914631586dc3a85b22d10db0620885a5ee83d0a560f5b67d5d674311fa7357`; the byte-preserving gzip copy is [native_first4_sse.log.gz](native_first4_sse.log.gz) (decompress with `gzip -dc`). [Capture time](native_first4_sse_time.txt) records the stop condition.

A separate native mesh probe used `mris_sphere -threads 4 -seed 1234 -w 1` on byte-identical input copies. It stopped on `lh.sphere0004` after 30.03 s (SIGTERM), leaving only initial and first four update files (`sphere0000`–`sphere0004`). [The snapshot manifest](snapshot_probe.json) records its command, input/binary hashes and five output hashes. Files remain in the gpucw1 scratch directory named in the manifest. The Python [four-step comparator](../experimental/probe_standard_sphere_default_continuous.py), SHA-256 `bbf2f475da6b06c2a2f70d82a436ca0be0a302ffab4e9b1ace1b340f0e621648`, independently rebuilt the metric/gradient/line search from the candidate files and compared the native checkpoints. [Its JSON](python_vs_native_first4.json) records all active implementation-file hashes, terms, decisions and same-index coordinates. It stopped at the first unequal update as designed. The Python `finish` module hash in that JSON is provenance only; cleanup was not run.

The ordinary full official candidate run did not save per-update snapshots. Thus identical input SHA and the historical frozen-official-input validation of `-w 1` do not directly prove that this **candidate** `-w 1` prefix is bitwise identical to its ordinary full run. The diagnostic `sses:` capture and mesh capture also ran as separate processes; their first-step choices agree to the printed precision.

## Earliest difference

| Candidate-input initial repair | Native printed | Python independent calculation |
| --- | ---: | ---: |
| Initial projected `sphere0000` coordinates | 319,866 components | 319,866/319,866 exact to native |
| Initial negative-area SSE | 275.835616 | 275.835615519 |
| Initial distance SSE | **26.863932** | **26.863064785** |
| Initial total SSE | **302.699548** | **302.698680304** |
| First selected line-search trial | index 2, dt 1027.938 | index 2, dt 1027.937440996 |
| First updated coordinates | 319,866 components | 308,844/319,866 exact; maximum vertex error 0.000011444 mm |

The first discrepancy is therefore in the distance objective **before the first optimizer update**. The initial mesh is exact, and the negative-area term agrees at its six printed decimal places; the distance term accounts for the displayed total-SSE gap. This narrows the next audit to native versus Python target metric distances or current spherical-distance/SSE evaluation on the same initial coordinates. It does not establish which of those two is responsible. A scalar fit of the Python first-step `dt` to the native checkpoint (1027.937658) still leaves a 0.000011444 mm maximum error, so changing only the step size is not a verified fix.

The first four native printed `dt` values were 1027.938, 18451.210, 399.497 and 34325.025. The independent full Python candidate run recorded 1027.937441, 18450.785169, 399.364756 and 34319.906153, with matching `initial_repair`/1024-average plan for indices 0–2 and 256 averages at index 3. Values after index 0 are **not controlled same-state comparisons** because the first updated coordinates already differ. No production code or numerical setting was changed.

Two unsuccessful option combinations are excluded from the numerical result: `-v` without a value printed help and exited before optimization; combining `DIAG` with `-w 1` exited with an `MRISwriteIntoVolume` diagnostic error after writing only `sphere0000`. Both failed before an accepted update. The successful captures above use the separate modes already established by the repository's sphere probes.
