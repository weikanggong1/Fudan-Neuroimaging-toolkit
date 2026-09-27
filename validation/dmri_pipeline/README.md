# dMRI parameter-map pipeline validation

The public report uses one real UKB-format AP/PA acquisition and the supplied
T1w image. It checks the shared TOPUP → EDDY → DTIFIT/NODDI prefix, the
UKB-style TBSS branch, and the T1-plus-tensor MMORF branch. The repository does
not contain the source images or a subject identifier.

The validation separates three questions:

1. whether both FNIT branches write the same nine names on the same MNI152
   1 mm grid;
2. how the TBSS branch differs from a fresh FSL 6.0.7.4 / UKB v1.5 run on the
   same native maps;
3. how the native parameter maps differ from the existing official UKB outputs
   after the whole correction and fitting prefix.

[`compare_tbss.py`](compare_tbss.py) computes standard-map and skeleton-map
metrics from matched output directories. [`compare_native.py`](compare_native.py)
separates common-support differences from mask/support differences in the nine
native maps. [`run_official_tbss.sh`](run_official_tbss.sh)
contains the exact forward FLIRT, three-stage FNIRT, `applywarp`, and skeleton
commands used for the timed oracle. The final aggregate is stored in
[`report.public.json`](report.public.json). The matched-native timing,
topology profile, fixed-affine, fixed-warp, and skeleton attribution are
summarized in [`tbss_diagnosis.public.json`](tbss_diagnosis.public.json).
