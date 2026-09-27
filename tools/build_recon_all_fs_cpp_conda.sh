#!/usr/bin/env bash
# Build selected FreeSurfer 8.2 C++ stages entirely with the active Conda toolchain.
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 FREESURFER_SOURCE OUTPUT_DIR" >&2
  exit 2
fi
source_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
: "${CONDA_PREFIX:?activate the build Conda environment first}"
for tool in cmake ninja; do
  command -v "$tool" >/dev/null || { echo "missing Conda build tool: $tool" >&2; exit 2; }
  case "$(realpath "$(command -v "$tool")")" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "$tool resolves outside CONDA_PREFIX" >&2; exit 2 ;;
  esac
done
for compiler in CC CXX FC; do
  binary="${!compiler:-}"
  test -n "$binary" && test -x "$binary" || {
    echo "Conda compiler variable $compiler is not set to an executable" >&2; exit 2;
  }
  case "$(realpath "$binary")" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "$compiler resolves outside CONDA_PREFIX" >&2; exit 2 ;;
  esac
done
if test -s "$source_dir/.fnit-source-commit"; then
  source_commit=$(cat "$source_dir/.fnit-source-commit")
else
  source_commit=$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || true)
fi
expected_commit=d932c45b7941662ea380a05efef580568b98d41a
if [[ "$source_commit" != "$expected_commit" ]]; then
  echo "FreeSurfer source commit must be $expected_commit; found $source_commit" >&2
  exit 2
fi
for file in LICENSE.txt CMakeLists.txt utils/CMakeLists.txt mri_em_register/CMakeLists.txt mris_fix_topology/CMakeLists.txt mris_make_surfaces/CMakeLists.txt mris_register/CMakeLists.txt mris_inflate/CMakeLists.txt mris_sphere/CMakeLists.txt resurf/Code/mris_multimodal_refinement.h; do
  test -s "$source_dir/$file" || { echo "missing source: $file" >&2; exit 2; }
done
itk_config=$(find "$CONDA_PREFIX/lib/cmake" -maxdepth 3 -name ITKConfig.cmake -print -quit)
test -n "$itk_config" || { echo "Conda ITKConfig.cmake not found" >&2; exit 2; }
# gpucw1 runs glibc 2.17; a 2.28 Conda sysroot links binaries that cannot start there.
if [[ "$(getconf GNU_LIBC_VERSION)" == "glibc 2.17" ]]; then
  sysroot_libc="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/lib64/libc.so.6"
  if [[ ! -L "$sysroot_libc" || "$(readlink "$sysroot_libc")" != "libc-2.17.so" ]]; then
    echo "Conda sysroot_linux-64=2.17 is required on this glibc 2.17 host" >&2
    exit 2
  fi
fi
mkdir -p "$output_dir/bin"
cp "$source_dir/LICENSE.txt" "$output_dir/FreeSurfer-LICENSE.txt"
build_source="$output_dir/source"
if ! test -s "$build_source/.fnit-conda-cmake-patched"; then
  test ! -e "$build_source" || { echo "partial build source exists: $build_source" >&2; exit 2; }
  cp -a "$source_dir" "$build_source"
  python3 - "$build_source/CMakeLists.txt" <<'PYTHON'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
old = 'set(PYTHON_EXECUTABLE "${FS_PACKAGES_DIR}/fspython/${FSPYTHON_VERSION}/bin/python${FSPYTHON_VERSION}")'
new = '# FNIT Conda port: honor the explicitly provided Conda Python.\n   if(NOT PYTHON_EXECUTABLE)\n      ' + old + '\n   endif()'
assert text.count(old) == 1
path.write_text(text.replace(old, new))
PYTHON
  printf 'Conda Python override only; upstream commit %s\n' "$source_commit" > "$build_source/.fnit-conda-cmake-patched"
fi
diff -u "$source_dir/CMakeLists.txt" "$build_source/CMakeLists.txt" > "$output_dir/conda-cmake.patch" || true
# The shared NFS server clock can predate archive mtimes and make Ninja reconfigure forever.
find "$build_source" -type f -exec touch -d '2000-01-01 00:00:00 UTC' {} +
unset FREESURFER_HOME
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
build_dir="$output_dir/build"
cmake -S "$build_source" -B "$build_dir" -G Ninja \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS:-} -Wno-error=restrict -Wno-error=format-overflow" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$output_dir" \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DFS_PACKAGES_DIR="$CONDA_PREFIX" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DITK_DIR="$(dirname "$itk_config")" \
  -DMINIMAL=ON -DBUILD_GUIS=OFF -DBUILD_ATTIC=OFF \
  -DDISABLE_LINEPROF=ON -DINFANT_MODULE=OFF -DQATOOLS_MODULE=OFF \
  -DDISTRIBUTE_FSPYTHON=OFF -DINSTALL_PYTHON_DEPENDENCIES=OFF \
  2>&1 | tee "$output_dir/configure.log"
targets=(mri_em_register mris_fix_topology mris_place_surface mris_register mris_inflate mris_sphere)
cmake --build "$build_dir" --parallel 4 --target "${targets[@]}" 2>&1 | tee "$output_dir/build.log"
for target in "${targets[@]}"; do
  binary=$(find "$build_dir" -type f -name "$target" -perm /111 -print -quit)
  test -n "$binary" || { echo "missing built binary: $target" >&2; exit 1; }
  install -m 755 "$binary" "$output_dir/bin/$target"
  ldd "$output_dir/bin/$target" > "$output_dir/$target.ldd"
  if grep -E '/public/software/apps/Freesurfer|/tmp/fs_itk_build' "$output_dir/$target.ldd"; then
    echo "$target links an external FreeSurfer/ITK installation" >&2
    exit 1
  fi
  timeout 15s "$output_dir/bin/$target" --help > "$output_dir/$target.launch.log" 2>&1 || true
  if grep -Eq 'GLIBC_[0-9.]+.*not found|error while loading shared libraries' "$output_dir/$target.launch.log"; then
    echo "$target cannot start on this host; inspect $output_dir/$target.launch.log" >&2
    exit 1
  fi
done
sha256sum "$output_dir"/bin/* > "$output_dir/bin.sha256"
conda list --explicit > "$output_dir/conda-explicit.txt"
printf 'SOURCE_COMMIT=%s\nCONDA_PREFIX=%s\n' "$source_commit" "$CONDA_PREFIX" > "$output_dir/build-provenance.txt"
