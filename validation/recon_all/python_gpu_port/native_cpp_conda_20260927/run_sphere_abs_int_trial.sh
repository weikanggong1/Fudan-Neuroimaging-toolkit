#!/usr/bin/env bash
set -euo pipefail

root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927
trial="$root/sphere_abs_int_trial_20260927"
pair="$root/sphere_registration_pair_20260927/sphere/official"
e2e=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_conda_cpp_cc_e2e_20260927

test "$(cat "$e2e/reports/e2e.status")" = 'exit=0'
test ! -e "$trial/run/subjects/sub01"
test -s "$trial/mris_sphere"
test -s "$pair/subjects/sub01/surf/lh.sphere"
: "${FS_LICENSE:?Set FS_LICENSE to a readable private license file}"
test -r "$FS_LICENSE"

mkdir -p "$trial/run/subjects"
cp -a "$pair/subjects/sub01" "$trial/run/subjects/sub01"
rm -f "$trial/run/subjects/sub01/surf/lh.sphere"
(cd "$trial/run/subjects/sub01" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$trial/input_manifest.sha256"
cmp "$pair/input_manifest.sha256" "$trial/input_manifest.sha256"

export SUBJECTS_DIR="$trial/run/subjects"
export FREESURFER_HOME="$root/assets"
export OMP_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
surf="$trial/run/subjects/sub01/surf"
cd "$trial/run/subjects/sub01/scripts"
printf '%q ' "$trial/mris_sphere" -threads 4 -seed 1234 "$surf/lh.inflated" "$surf/lh.sphere" > "$trial/command.txt"
printf '\n' >> "$trial/command.txt"
/usr/bin/time -f '%e' -o "$trial/wall_seconds.txt" timeout 720 \
  "$trial/mris_sphere" -threads 4 -seed 1234 \
  "$surf/lh.inflated" "$surf/lh.sphere" > "$trial/command.log" 2>&1
sha256sum "$trial/mris_sphere" "$root/fs_cpp/bin/mris_sphere" \
  "$surf/lh.inflated" "$surf/lh.sphere" \
  "$pair/subjects/sub01/surf/lh.sphere" > "$trial/output.sha256"

"$root/conda_env/bin/python" - "$trial" "$pair" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import nibabel.freesurfer.io as fsio
import numpy as np

trial, pair = map(Path, sys.argv[1:])
candidate = trial / 'run/subjects/sub01/surf/lh.sphere'
reference = pair / 'subjects/sub01/surf/lh.sphere'
xyz_a, faces_a = fsio.read_geometry(str(candidate))
xyz_b, faces_b = fsio.read_geometry(str(reference))
same_shape = xyz_a.shape == xyz_b.shape
same_faces = faces_a.shape == faces_b.shape and bool(np.array_equal(faces_a, faces_b))
result = {
    'candidate_command': (trial / 'command.txt').read_text().strip(),
    'wall_seconds': float((trial / 'wall_seconds.txt').read_text()),
    'input_manifest_sha256': hashlib.sha256((trial / 'input_manifest.sha256').read_bytes()).hexdigest(),
    'reference_input_manifest_sha256': hashlib.sha256((pair / 'input_manifest.sha256').read_bytes()).hexdigest(),
    'trial_binary_sha256': hashlib.sha256((trial / 'mris_sphere').read_bytes()).hexdigest(),
    'production_binary_sha256': hashlib.sha256((trial.parent / 'fs_cpp/bin/mris_sphere').read_bytes()).hexdigest(),
    'candidate_output_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
    'official_output_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
    'candidate_vertices': list(xyz_a.shape),
    'official_vertices': list(xyz_b.shape),
    'candidate_faces': list(faces_a.shape),
    'official_faces': list(faces_b.shape),
    'ordered_faces_exact': same_faces,
    'ordered_coordinates_exact': same_shape and bool(np.array_equal(xyz_a, xyz_b)),
}
if same_shape:
    absdiff = np.abs(xyz_a - xyz_b)
    dist = np.linalg.norm(xyz_a - xyz_b, axis=1)
    result.update({
        'coordinate_scalars_different': int(np.count_nonzero(absdiff)),
        'coordinate_max_abs_mm': float(absdiff.max()),
        'vertex_distance_mean_mm': float(dist.mean()),
        'vertex_distance_median_mm': float(np.median(dist)),
        'vertex_distance_p95_mm': float(np.percentile(dist, 95)),
        'vertex_distance_p99_mm': float(np.percentile(dist, 99)),
        'vertex_distance_max_mm': float(dist.max()),
        'vertices_over_1mm': int(np.count_nonzero(dist > 1.0)),
    })
result['first_scale_line'] = next((line for line in (trial / 'command.log').read_text(errors='replace').splitlines() if 'scaling brain by' in line), None)
(trial / 'accuracy.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2, sort_keys=True))
PY
