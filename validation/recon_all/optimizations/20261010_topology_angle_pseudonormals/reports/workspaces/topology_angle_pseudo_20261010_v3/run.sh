#!/bin/bash
set -uo pipefail
source <FNIT_ROOT>/logs/recon_runtime_env_20261009.sh
export PYTHONPATH=<FNIT_ROOT>/workspaces/topology_angle_pseudo_20261010_v3/src:<FNIT_ROOT>/workspaces/topology_angle_pseudo_20261010_v3/tools
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export NUMBA_CACHE_DIR=<FNIT_ROOT>/workspaces/topology_angle_pseudo_20261010_v3/numba-cache
taskset -c 120-123 "$FNIT_RECON_ENV/bin/python" <FNIT_ROOT>/workspaces/topology_angle_pseudo_20261010_v3/driver.py > <FNIT_ROOT>/runs/topology_angle_pseudo_20261010_v3/driver.log 2>&1
s=$?
echo "$s" > <FNIT_ROOT>/runs/topology_angle_pseudo_20261010_v3/driver.exitcode
exit "$s"
