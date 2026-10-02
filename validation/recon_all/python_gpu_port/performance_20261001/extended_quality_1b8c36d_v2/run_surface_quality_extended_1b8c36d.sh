#!/usr/bin/env bash
# headcw 的独立只读 CPU benchmark；不触碰整例、GPU 或生产快照。
set -euo pipefail
task_root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929
volume_root="$task_root/volume_parity_20260930"
code_root="$volume_root/fixes_20261001/code_1b8c36d" # 仅导入冻结成熟检测组件
diagnostic_root="$volume_root/fixes_20261001/extended_quality_1b8c36d_v2" # 正长度重叠判据的独立目录
python_executable="$task_root/fnit_main_env/bin/python" # 声明 Conda 环境
export PYTHONPATH="$code_root/src" CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export NUMBA_CACHE_DIR="$diagnostic_root/numba_cache" # JIT 缓存也只写诊断目录
for case_name in sub01 sub02; do
  subject_directory="$volume_root/full_sub02_1b8c36d"
  if [ "$case_name" = sub01 ]; then subject_directory="$volume_root/full_sub01_1b8c36d_retry1"; fi
  /usr/bin/time -p "$python_executable" "$diagnostic_root/benchmark_surface_quality_extended.py" \
    --subject "$subject_directory" --output "$diagnostic_root/$case_name" \
    --code-version 1b8c36d25a68e253a1e59b6d02114890afa467de \
    --threads 4 --cross-timeout-seconds 180 --max-bbox-pairs 20000000 \
    > "$diagnostic_root/$case_name.log" 2>&1
done
