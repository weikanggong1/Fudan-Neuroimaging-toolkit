#!/bin/bash
set -uo pipefail
source <FNIT_ROOT>/logs/recon_runtime_env_20261009.sh
export PYTHONPATH=<FNIT_ROOT>/workspaces/topology_candidate_normals_20261010_v2/src:<FNIT_ROOT>/workspaces/topology_candidate_normals_20261010_v2/tools
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export NUMBA_CACHE_DIR=<FNIT_ROOT>/workspaces/topology_candidate_normals_20261010_v2/numba-cache
taskset -c 120-123 "$FNIT_RECON_ENV/bin/python" <FNIT_ROOT>/workspaces/topology_candidate_normals_20261010_v2/driver.py > <FNIT_ROOT>/runs/topology_candidate_normals_20261010_v2/driver.log 2>&1
s=$?
echo "$s" > <FNIT_ROOT>/runs/topology_candidate_normals_20261010_v2/driver.exitcode
exit "$s"
