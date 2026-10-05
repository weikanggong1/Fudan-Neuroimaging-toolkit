#!/bin/bash
# Independent benchmark build. No production source or installed prefix edits.
set -euo pipefail
FNIT_SERVER_ROOT=/cwStorage/home/gongwk/Notebook_code/FNIT
PCG_WORK="$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261004/synth/fnirt-pcg-shared-v2"
PCG_OLD="$FNIT_SERVER_ROOT/workspaces/smri_cpu_20261004/remaining_20261004/gems/fnirt-first-diff-v2"
FSL_REFERENCE_ROOT=/public/software/apps/FSL/6.0.7.4
cd "$PCG_WORK"
g++ -std=c++17 -O0 -fPIC -I"$FSL_REFERENCE_ROOT/include" -I"$FSL_REFERENCE_ROOT/include/newmat" \
    -I"$FSL_REFERENCE_ROOT/src/fsl-fnirt" -c official_shared.cpp -o official_shared.o >build-object.log 2>&1
g++ official_shared.o "$PCG_OLD/fnirt_costfunctions.o" "$PCG_OLD/fnirtfns.o" \
    "$PCG_OLD/intensity_mappers.o" "$PCG_OLD/matching_points.o" \
    -L"$FSL_REFERENCE_ROOT/lib" -Wl,-rpath,"$FSL_REFERENCE_ROOT/lib" \
    -lfsl-warpfns -lfsl-meshclass -lfsl-basisfield -lfsl-newimage -lfsl-miscmaths \
    -lfsl-utils -lfsl-znz -lfsl-NewNifti -lfsl-cprob -lz -llapack -lblas -lpthread \
    -o official_shared >build-link.log 2>&1
printf 'built\n'
