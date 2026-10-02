#!/bin/bash
set -euo pipefail
python -c 'import pathlib,time,json
r=pathlib.Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001")
while not (r/'\''comparison.retry2.finished'\'').exists():
 p=r/'\''whole_comparison_retry2/progress.json'\''
 try:
  if p.exists() and json.loads(p.read_text()).get('\''status'\'')=='\''failed'\'': raise SystemExit('\''comparison failed; diagnostic not started'\'')
 except json.JSONDecodeError: pass
 time.sleep(15)'
ssh -S /tmp/fnit-gpucw1-gongwk@gpucw1:22 -oBatchMode=yes gongwk@gpucw1 'env PYTHONPATH='\''/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/candidate5b/src'\'' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 FS_LICENSE='\''/cwStorage/home/gongwk/.config/freesurfer-codex/license.txt'\'' '\''/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python'\'' '\''/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/diagnose_white_preaparc_drift.py'\'' --baseline-subject '\''/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/volume_parity_20260930/hotspots_20261001/full_sub01_hotspots_c248520_retry1'\'' --candidate-subject '\''/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_sub01_candidate_retry1'\'' --output '\''/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/white_preaparc_drift_sub01_rh'\'' --binary '\''/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/native_bundle/bin/mris_place_surface'\'' --assets '\''/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/assets'\'' --code-commit '\''61926c7dceae2f9097fa296e306cf1efa0a09191'\'' --hemi rh --threads 4'
