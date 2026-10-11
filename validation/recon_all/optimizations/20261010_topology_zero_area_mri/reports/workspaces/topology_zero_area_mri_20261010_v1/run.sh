#!/bin/bash
set -uo pipefail
source <FNIT_ROOT>/logs/recon_runtime_env_20261009.sh
export PYTHONPATH=<FNIT_ROOT>/workspaces/topology_zero_area_mri_20261010_v1/src:<FNIT_ROOT>/workspaces/topology_zero_area_mri_20261010_v1/tools
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export NUMBA_CACHE_DIR=<FNIT_ROOT>/workspaces/topology_zero_area_mri_20261010_v1/numba-cache
taskset -c 120-123 "$FNIT_RECON_ENV/bin/python" <FNIT_ROOT>/workspaces/topology_zero_area_mri_20261010_v1/driver.py > <FNIT_ROOT>/runs/topology_zero_area_mri_20261010_v1/driver.log 2>&1
s=$?
echo "$s" > <FNIT_ROOT>/runs/topology_zero_area_mri_20261010_v1/driver.exitcode
exit "$s"
