#!/usr/bin/env bash
# Install FNIT's pinned source-built recon-all programs into the active Conda env.
set -euo pipefail
: "${CONDA_PREFIX:?activate the FNIT Conda environment first}"
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
  git -C "$source_dir" init -q
  if git -C "$source_dir" fetch -q --depth 1 https://github.com/freesurfer/freesurfer.git "$commit"; then
    git -C "$source_dir" checkout -q --detach FETCH_HEAD
  else
    rm -rf "$source_dir/.git"
    archive=$(mktemp "$CONDA_PREFIX/share/fnit/freesurfer.XXXXXX.tar.gz")
    trap 'rm -f "$archive"' EXIT
    curl -fL --retry 3 --connect-timeout 20 \
      "https://codeload.github.com/freesurfer/freesurfer/tar.gz/$commit" -o "$archive"
    expected_archive=2e76f40415f3e334b6fcd2ce548b451e9219bc9ffaecc46e752d4853e13aff0a
    actual_archive=$(sha256sum "$archive" | cut -d' ' -f1)
    [[ "$actual_archive" == "$expected_archive" ]] || {
      echo "FreeSurfer source archive SHA-256 mismatch" >&2; exit 2;
    }
    tar -xzf "$archive" --strip-components=1 -C "$source_dir"
    printf '%s\n' "$commit" > "$source_dir/.fnit-source-commit"
    rm -f "$archive"
    trap - EXIT
  fi
fi
bash "$repo_root/tools/build_recon_all_fs_cpp_conda.sh" "$source_dir" "$build_dir"
for source in "$build_dir"/bin/*; do
  name=${source##*/}
  install -m 755 "$source" "$CONDA_PREFIX/bin/$name"
  patchelf --set-rpath '$ORIGIN/../lib' "$CONDA_PREFIX/bin/$name"
  if ldd "$CONDA_PREFIX/bin/$name" | grep -E 'not found|/Freesurfer/|/freesurfer/'; then
    echo "unexpected dynamic dependency in $name" >&2
    exit 1
  fi
done
sha256sum "$CONDA_PREFIX"/bin/{fnit_n4_itk,mri_em_register,mri_segment,mri_edit_wm_with_aseg,mris_fix_topology_fnit,mris_remove_intersection,mris_inflate,mris_place_surface,mrisp_paint,mris_curvature_stats,mri_label2vol,mri_warp_convert,mri_ca_register,mri_convert} \
  > "$build_dir/installed-bin.sha256"
printf 'Installed FNIT recon-all native stages in %s/bin\n' "$CONDA_PREFIX"
