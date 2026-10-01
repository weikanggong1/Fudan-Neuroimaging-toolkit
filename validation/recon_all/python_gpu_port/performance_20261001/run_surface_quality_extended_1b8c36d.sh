#!/usr/bin/env bash
# headcw 的独立只读 CPU benchmark；不触碰整例、GPU 或生产快照。
set -euo pipefail
task_root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
volume_root="$task_root/volume_parity_20260930"
code_root="$volume_root/fixes_20261001/code_1b8c36d" # 仅导入冻结成熟检测组件
diagnostic_root="$volume_root/fixes_20261001/extended_quality_1b8c36d_v3" # 同判据，追加几何细节和只读对照
python_executable="$task_root/fnit_main_env/bin/python" # 声明 Conda 环境
export PYTHONPATH="$code_root/src" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export NUMBA_CACHE_DIR="$diagnostic_root/numba_cache" # JIT 缓存也只写诊断目录
for case_name in candidate_sub01 candidate_sub02 legacy_sub01 legacy_sub02 official_sub01 official_sub02; do
  source_kind=fnit
  code_version=1b8c36d25a68e253a1e59b6d02114890afa467de
  case "$case_name" in
    candidate_sub01) subject_directory="$volume_root/full_sub01_1b8c36d_retry1" ;;
    candidate_sub02) subject_directory="$volume_root/full_sub02_1b8c36d" ;;
    legacy_sub01) subject_directory="$volume_root/full_sub01_e036f57_uuid"; code_version=e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68 ;;
    legacy_sub02) subject_directory="$volume_root/full_sub02_e036f57"; code_version=e036f57b62b99d2af4cd8853ab2f1e6d2a9f8c68 ;;
    official_sub01) subject_directory=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_benchmark_pair_ac_20260924/official_subjects/a_official; code_version=existing-official-sub01-reference; source_kind=official ;;
    official_sub02) subject_directory="$task_root/reference_sub02_official/subjects/b_official"; code_version=existing-official-sub02-reference; source_kind=official ;;
  esac
  /usr/bin/time -p "$python_executable" "$diagnostic_root/benchmark_surface_quality_extended.py" \
    --subject "$subject_directory" --output "$diagnostic_root/$case_name" \
    --code-version "$code_version" --source-kind "$source_kind" \
    --threads 4 --cross-timeout-seconds 180 --max-bbox-pairs 20000000 \
    > "$diagnostic_root/$case_name.log" 2>&1
done
