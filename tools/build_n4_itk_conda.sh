#!/usr/bin/env bash
# Build the in-repository N4 CPU stage with Conda ITK and compiler packages.
set -euo pipefail
if (( $# != 1 )); then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi
: "${CONDA_PREFIX:?activate the recon-all Conda environment first}"
: "${CXX:?activate the Conda C++ compiler first}"
for tool in cmake ninja; do
  case "$(realpath "$(command -v "$tool")")" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "$tool must resolve inside CONDA_PREFIX" >&2; exit 2 ;;
  esac
done
itk_config=$(find "$CONDA_PREFIX/lib/cmake" -maxdepth 3 -name ITKConfig.cmake -print -quit)
test -n "$itk_config" || { echo "Conda ITKConfig.cmake not found" >&2; exit 2; }
case "$(realpath "$CXX")" in
  "$CONDA_PREFIX"/*) ;;
  *) echo "CXX must resolve inside CONDA_PREFIX" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "$0")/.." && pwd)
output_dir=$(realpath -m "$1")
cmake -S "$root/tools/n4_itk" -B "$output_dir/n4-build" -G Ninja \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DITK_DIR="$(dirname "$itk_config")" \
  -DCMAKE_INSTALL_PREFIX="$output_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$output_dir/n4-build" --parallel 2
cmake --install "$output_dir/n4-build"
ldd "$output_dir/bin/fnit_n4_itk" > "$output_dir/fnit_n4_itk.ldd"
if grep -E 'not found|/public/software/apps/Freesurfer' "$output_dir/fnit_n4_itk.ldd"; then
  echo "N4 binary has a missing or external FreeSurfer dynamic dependency" >&2
  exit 1
fi
