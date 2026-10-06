#!/usr/bin/env bash
# Install FNIT's pinned source-built recon-all programs into the active Conda env.
set -euo pipefail
: "${CONDA_PREFIX:?activate the FNIT Conda environment first}"
for tool in python patchelf; do
  test -x "$CONDA_PREFIX/bin/$tool" || { echo "missing Conda tool: $tool" >&2; exit 2; }
done
repo_root=$(cd "$(dirname "$0")/.." && pwd)
source_dir=${FNIT_RECON_ALL_SOURCE:-$CONDA_PREFIX/share/fnit/recon_all_fs_source_d932c45_full}
build_dir="$CONDA_PREFIX/share/fnit/recon_all_native_full"
commit=d932c45b7941662ea380a05efef580568b98d41a
if [[ ! -d "$source_dir/.git" && ! -f "$source_dir/.fnit-source-commit" ]]; then
  if [[ -n "${FNIT_RECON_ALL_SOURCE:-}" ]]; then
    echo "FNIT_RECON_ALL_SOURCE must name a validated source tree" >&2
    exit 2
  fi
  mkdir -p "$source_dir"
  expected_archive=2e76f40415f3e334b6fcd2ce548b451e9219bc9ffaecc46e752d4853e13aff0a
  archive=$(mktemp "$CONDA_PREFIX/share/fnit/freesurfer.XXXXXX.tar.gz")
  trap 'rm -f "$archive" "$archive.part"' EXIT
  if "$CONDA_PREFIX/bin/python" - "$repo_root/src" "$archive" "$expected_archive" <<'PYTHON'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fnit._release_assets import release_asset_metadata, release_url_for
from fnit.recon_all.assets import _download_verified
archive, digest = Path(sys.argv[2]), sys.argv[3]
metadata = release_asset_metadata(digest)
release_url = release_url_for(digest)
if metadata is None or release_url is None:
    raise SystemExit(3)
_download_verified(release_url, archive, metadata["size"], digest)
print("Verified FNIT Release native source archive")
PYTHON
  then
    tar -xzf "$archive" --strip-components=1 -C "$source_dir"
    printf '%s\n' "$commit" > "$source_dir/.fnit-source-commit"
  else
    # A mirror failure keeps the pinned Git/codeload source paths available.
    rm -f "$archive" "$archive.part"
    git -C "$source_dir" init -q
    if git -C "$source_dir" fetch -q --depth 1 https://github.com/freesurfer/freesurfer.git "$commit"; then
      git -C "$source_dir" checkout -q --detach FETCH_HEAD
    else
      rm -rf "$source_dir/.git"
      curl -fL --retry 3 --connect-timeout 20 \
        "https://codeload.github.com/freesurfer/freesurfer/tar.gz/$commit" -o "$archive"
      actual_archive=$(sha256sum "$archive" | cut -d' ' -f1)
      [[ "$actual_archive" == "$expected_archive" ]] || {
        echo "FreeSurfer source archive SHA-256 mismatch" >&2; exit 2;
      }
      tar -xzf "$archive" --strip-components=1 -C "$source_dir"
      printf '%s\n' "$commit" > "$source_dir/.fnit-source-commit"
    fi
  fi
  rm -f "$archive" "$archive.part"
  trap - EXIT
fi
bash "$repo_root/tools/build_recon_all_fs_cpp_conda.sh" "$source_dir" "$build_dir"
for source in "$build_dir"/bin/*; do
  name=${source##*/}
  install -m 755 "$source" "$CONDA_PREFIX/bin/$name"
  "$CONDA_PREFIX/bin/patchelf" --set-rpath '$ORIGIN/../lib' "$CONDA_PREFIX/bin/$name"
  if ldd "$CONDA_PREFIX/bin/$name" | grep -E 'not found|/Freesurfer/|/freesurfer/'; then
    echo "unexpected dynamic dependency in $name" >&2
    exit 1
  fi
done
sha256sum "$CONDA_PREFIX"/bin/{fnit_n4_itk,mri_em_register,mri_segment,mri_edit_wm_with_aseg,mris_fix_topology_fnit,mris_remove_intersection,mris_inflate,mris_place_surface,mris_place_surface_white_fast,mrisp_paint,mris_curvature_stats,mri_label2vol,mri_warp_convert,mri_ca_register,mri_convert,mris_expand} \
  > "$build_dir/installed-bin.sha256"
"$CONDA_PREFIX/bin/python" - "$build_dir" "$CONDA_PREFIX" <<'PYTHON'
import hashlib, json, os, subprocess, sys
from pathlib import Path
build, prefix = map(Path, sys.argv[1:])
manifest = json.loads((build / "native-optimizations.json").read_text())
for name, item in manifest["programs"].items():
    binary = prefix / "bin" / name
    item["build_binary_sha256"] = item["sha256"]
    item["path"] = str(binary)
    item["sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    if name == "mri_em_register":
        env = dict(os.environ, FNIT_GCA_QUERY_CAPABILITIES="1")
        actual = json.loads(subprocess.check_output([str(binary)], env=env, text=True, timeout=15))
    elif name == "mris_place_surface_white_fast":
        actual = json.loads(subprocess.check_output(
            [str(binary), "--fnit-placement-capabilities"], text=True, timeout=15))
    else:
        continue
    if actual != item["capabilities"]:
        raise ValueError("installed capability differs: " + name)
manifest["installed_conda_prefix"] = str(prefix)
(build / "installed-native-optimizations.json").write_text(json.dumps(manifest, indent=2) + "\n")
PYTHON
printf 'Installed FNIT recon-all native stages in %s/bin\n' "$CONDA_PREFIX"
