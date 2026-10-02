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
repo_root=$(cd "$(dirname "$0")/.." && pwd)
test -x "$CONDA_PREFIX/bin/python" || { echo "missing Conda Python" >&2; exit 2; }
for helper in src/fnit/recon_all/mri_em_register_native_build.py src/fnit/recon_all/mri_em_register_native_score.hpp tools/place_surface_hotspots/build_native.py; do
  test -s "$repo_root/$helper" || { echo "missing FNIT native patch helper: $helper" >&2; exit 2; }
done
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
expected_commit=d932c45b7941662ea380a05efef580568b98d41a
if git -C "$source_dir" rev-parse --show-toplevel >/dev/null 2>&1; then
  source_commit=$(git -C "$source_dir" rev-parse HEAD)
  source_validation=clean-git
  if [[ "$(realpath "$(git -C "$source_dir" rev-parse --show-toplevel)")" != "$source_dir" ]] ||
     [[ -n "$(git -C "$source_dir" status --porcelain --untracked-files=all)" ]]; then
    echo "FreeSurfer source Git checkout must be clean and rooted at $source_dir" >&2
    exit 2
  fi
elif test -s "$source_dir/.fnit-source-commit"; then
  source_commit=$(cat "$source_dir/.fnit-source-commit")
  # The nonlinear stage needs the complete GitHub codeload source archive.
  expected_tree=313afb62ea5b5c7d5aa9d78659403b126c63e7cdd91465391ce6a2138c93693c
  source_tree=$(cd "$source_dir" && find . -type f ! -name .fnit-source-commit -print0 |
    LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
  if [[ "$source_tree" != "$expected_tree" ]]; then
    echo "FreeSurfer source archive tree differs from the validated commit" >&2
    exit 2
  fi
  source_validation="archive-tree-sha256:$source_tree"
else
  echo "FreeSurfer source must be a clean Git checkout or validated source archive" >&2
  exit 2
fi
if [[ "$source_commit" != "$expected_commit" ]]; then
  echo "FreeSurfer source commit must be $expected_commit; found $source_commit" >&2
  exit 2
fi
for file in LICENSE.txt CMakeLists.txt utils/CMakeLists.txt mri_em_register/CMakeLists.txt mris_fix_topology/CMakeLists.txt mris_make_surfaces/CMakeLists.txt mris_inflate/CMakeLists.txt mri_segment/CMakeLists.txt mri_edit_wm_with_aseg/CMakeLists.txt mri_warp_convert/CMakeLists.txt resurf/Code/mris_multimodal_refinement.h; do
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
  "$CONDA_PREFIX/bin/python" - "$build_source/CMakeLists.txt" <<'PYTHON'
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
# mri_warp_convert is excluded by FreeSurfer's MINIMAL target list.
if ! grep -q FNIT_RECON_WARP_TARGET "$build_source/CMakeLists.txt"; then
  cat >> "$build_source/CMakeLists.txt" <<'CMAKE'

# FNIT_RECON_WARP_TARGET
if(MINIMAL)
  add_subdirectory(mri_warp_convert)
endif()
CMAKE
fi
diff -u "$source_dir/CMakeLists.txt" "$build_source/CMakeLists.txt" > "$output_dir/conda-cmake.patch" || true
# Patch only this installer's source copy. The complete native optimizer stays
# in mri_em_register; its scorer is selected explicitly by FNIT_GCA_SCORER.
"$CONDA_PREFIX/bin/python" - "$repo_root" "$build_source" "$output_dir/gca-search-patch.json" <<'PYTHON'
import importlib.util, json, sys
from pathlib import Path
repo, source, report_path = map(Path, sys.argv[1:])
path = repo / "src/fnit/recon_all/mri_em_register_native_build.py"
spec = importlib.util.spec_from_file_location("fnit_gca_install_patch", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report = module.apply_native_search_patch(source)
report["install_patch_version"] = "gca-cachev3-capability2"
report["live_patcher_sha256"] = module.sha(path)
report_path.write_text(json.dumps(report, indent=2) + "\n")
PYTHON
# The shared NFS server clock can predate archive mtimes and make Ninja reconfigure forever.
find "$build_source" -type f -exec touch -d '2000-01-01 00:00:00 UTC' {} +
unset FREESURFER_HOME
export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
build_dir="$output_dir/build"
# Source mtimes above are deliberately normalized for NFS. Recompile this one
# installer-owned object so an older build cannot miss the newly applied patch.
rm -f "$build_dir/mri_em_register/CMakeFiles/mri_em_register.dir/emregisterutils.cpp.o"
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
targets=(mri_em_register mri_segment mri_edit_wm_with_aseg mris_fix_topology mris_remove_intersection mris_inflate mris_place_surface mrisp_paint mris_curvature_stats mri_label2vol mri_warp_convert mri_ca_register mri_convert)
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
# Recompile only the placement utility and capability-query main. Link them
# against an independent archive copy; the original placement binary and shared
# libutils.a remain available for pial and the other native targets.
"$CONDA_PREFIX/bin/python" - "$repo_root" "$build_dir" "$build_source" "$output_dir" "$CONDA_PREFIX" <<'PYTHON'
import importlib.util, json, shlex, subprocess, sys
from pathlib import Path
repo, build, source, output, prefix = map(Path, sys.argv[1:])
path = repo / "tools/place_surface_hotspots/build_native.py"
spec = importlib.util.spec_from_file_location("fnit_white_install_patch", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inputs = {name: module.sha(source / name) for name in module.EXPECTED}
if inputs != module.EXPECTED:
    raise ValueError("fixed placement source SHA-256 mismatch")
candidate_dir = output / "placement-white-fast-build"
binary = candidate_dir / "mris_place_surface_fnit_hotspot"
if candidate_dir.exists():
    report = json.loads((candidate_dir / "build.json").read_text())
    if (not report["patched"] or report["builder_sha256"] != module.sha(path)
            or report["input_source_sha256"] != inputs
            or report["original_archive_sha256"] != module.sha(build / "utils/libutils.a")
            or report["binary_sha256"] != module.sha(binary)):
        raise ValueError("existing white build differs; use a new OUTPUT_DIR")
else:
    commands = subprocess.check_output(
        [str(prefix / "bin/ninja"), "-t", "commands", "mris_place_surface"],
        cwd=build, text=True).splitlines()
    for original in (source / "utils/mrisurf_mri.cpp",
                     source / "mris_make_surfaces/mris_place_surface.cpp"):
        line = next(x for x in commands if " -c " in x and x.rstrip().endswith(str(original)))
        argv = shlex.split(line)
        if any(x in argv for x in ("&&", ";", "|")):
            raise ValueError("unexpected placement compile shell command")
        Path(argv[0]).resolve().relative_to(prefix.resolve())
    module.build_one(build, source, candidate_dir, commands, patched=True)
    report = json.loads((candidate_dir / "build.json").read_text())
capabilities = json.loads(subprocess.check_output(
    [str(binary), "--fnit-placement-capabilities"], text=True, timeout=15))
if capabilities != module.CAPABILITIES:
    raise ValueError("white placement capability mismatch")
report.update(install_patch_version="white-unused-face-mht-v1",
              installed_name="mris_place_surface_white_fast",
              intended_stages=["white_preaparc", "final_white"],
              capabilities=capabilities)
(output / "white-placement-patch.json").write_text(json.dumps(report, indent=2) + "\n")
PYTHON
install -m 755 "$output_dir/placement-white-fast-build/mris_place_surface_fnit_hotspot" \
  "$output_dir/bin/mris_place_surface_white_fast"
ldd "$output_dir/bin/mris_place_surface_white_fast" > "$output_dir/mris_place_surface_white_fast.ldd"
if grep -E 'not found|/public/software/apps/Freesurfer|/tmp/fs_itk_build' "$output_dir/mris_place_surface_white_fast.ldd"; then
  echo "white placement variant has an unavailable or external dependency" >&2
  exit 1
fi
# Build the topology variant that consumes the Python exact-centered sphere and
# pins the older FreeSurfer double-tanh and double-sqrt behavior. Keep the unmodified binary
# above for diagnostic comparisons only.
"$CONDA_PREFIX/bin/python" "$(dirname "$0")/build_recon_all_topology_fnit.py" \
  --build "$build_dir" --source "$build_source" \
  --output-dir "$output_dir/topology-fnit-build"
install -m 755 "$output_dir/topology-fnit-build/mris_fix_topology_fnit" \
  "$output_dir/bin/mris_fix_topology_fnit"
ldd "$output_dir/bin/mris_fix_topology_fnit" > "$output_dir/mris_fix_topology_fnit.ldd"
if grep -E '/public/software/apps/Freesurfer|/tmp/fs_itk_build' "$output_dir/mris_fix_topology_fnit.ldd"; then
  echo "patched topology binary links an external FreeSurfer/ITK installation" >&2
  exit 1
fi
bash "$(dirname "$0")/build_n4_itk_conda.sh" "$output_dir"
sha256sum "$output_dir"/bin/* > "$output_dir/bin.sha256"
conda list --explicit > "$output_dir/conda-explicit.txt"
printf 'SOURCE_COMMIT=%s\nSOURCE_VALIDATION=%s\nCONDA_PREFIX=%s\n' "$source_commit" "$source_validation" "$CONDA_PREFIX" > "$output_dir/build-provenance.txt"
"$CONDA_PREFIX/bin/python" - "$output_dir" "$source_commit" "$source_validation" "$repo_root" <<'PYTHON'
import hashlib, json, os, subprocess, sys
from pathlib import Path
output, commit, validation, repo = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
environment = dict(os.environ, FNIT_GCA_QUERY_CAPABILITIES="1")
gca = json.loads(subprocess.check_output([str(output / "bin/mri_em_register")],
                                       env=environment, text=True, timeout=15))
if (gca.get("version") != 2 or not gca.get("full_native_em")
        or not gca.get("fnit_gca_cached_search")
        or not gca.get("single_input_uint8")
        or gca.get("activation") != "FNIT_GCA_SCORER=cpu_cached"
        or gca.get("reduction") != "upstream_ROMP_partials"):
    raise ValueError("GCA cachev3 capability2 with upstream ROMP partials is required")
gca_patch = json.loads((output / "gca-search-patch.json").read_text())
white_patch = json.loads((output / "white-placement-patch.json").read_text())
programs = {}
for name, stages, patch, capabilities in (
    ("mri_em_register", ["complete_gca_registration"], gca_patch, gca),
    ("mris_place_surface_white_fast", ["white_preaparc", "final_white"],
     white_patch, white_patch["capabilities"]),
    ("mris_place_surface", ["pial", "surface_metrics"], None, None),
):
    binary = output / "bin" / name
    programs[name] = dict(path=str(binary), sha256=sha(binary), stages=stages,
                          patch=patch, capabilities=capabilities)
manifest = dict(schema_version=1, install_patch_version="native-hotspots-20261002-v1",
                upstream_commit=commit, source_validation=validation,
                license_sha256=sha(output / "FreeSurfer-LICENSE.txt"),
                installer_sha256={name: sha(repo / "tools" / name) for name in
                    ("build_recon_all_fs_cpp_conda.sh", "setup_recon_all_native_conda.sh")},
                programs=programs)
(output / "native-optimizations.json").write_text(json.dumps(manifest, indent=2) + "\n")
PYTHON
