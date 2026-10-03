# 官方 recon-all 来源：实际新目录单例接入

## 1. 范围

在 nodecw10 使用公开 ds001226 的 CON01 同一份原始 T1，调用 `prepare_recon_subject(recon_backend="freesurfer")` 重建解剖目录。它只检验官方来源适配器的真实运行、完成检查与再次调用跳过，不替代十人的 provided subject→SC benchmark，也不比较 FNIT 与官方 recon-all 数值精度。

运行限制为 CPU、4 个实际 CPU 的亲和性、`CUDA_VISIBLE_DEVICES=''`。官方调用依照适配器使用 `-all -parallel -openmp 4`。进程亲和性覆盖所有后代，防止双半球并行时超出四个实际 CPU。

## 2. 首次失败与修复

| 阶段 | 冻结 FNIT 源码 | 实际结果 |
| --- | --- | --- |
| 首次调用 | `8bc337c4ec688e64385ef5c2300d369281bf1a02` | 官方程序立即返回 `FREESURFER: Undefined variable.`；没有重建结果和完成缓存。 |
| 初始化修复 | `3b697657919347281d02285b3cd634008635a4c0` | 三个相关测试文件共55项通过，11.94秒；原官方 setup 在 nodecw10 的只读初始化检查成功。 |
| `official_retry_v2` | 同上 | 2026-10-03 14:36:34–16:17:46 UTC 实际完成；首次调用6069.7609秒，第二次1.02243秒返回 `skipped`，文件未改变、官方完成与格式/几何检查通过。 |

首次失败公开元数据见 [initial_failure.public.json](initial_failure.public.json)，修复测试输出见 [setup_fix_tests.txt](setup_fix_tests.txt)。MRI、完整 raw SHA、官方命令、PID与全部输出哈希只保存在服务器私有记录，不转移至本公开报告。最终公开摘要见 [fresh_complete.public.json](fresh_complete.public.json)；回收采用字段白名单。

根因是来源适配器以前只设置 `FREESURFER_HOME` 与 PATH，官方8.2脚本还消费完整 setup 中定义的变量。修复不猜测补单一变量：独立 Bash 子进程 source `SetUpFreeSurfer.sh`，再 exec 独立 argv；MRI 路径不插入 shell 字符串。两个 setup 脚本 SHA-256 纳入执行身份，变更导致解剖缓存失效。源目录没有移动；旧的 `frozen_v1`、首次失败和其他十人作业不变。

## 3. 执行和验收

可复用脚本为 [run_backend.py](run_backend.py)。许可证只检查可读性并通过环境供官方程序使用，不读取、复制或输出其内容。

```bash
FNIT_SOURCE_DIR=/path/to/frozen/source
FNIT_PYTHON=/path/to/fnit/conda/bin/python
RAW_T1=/path/to/public/sub-CON01_T1w.nii.gz
FREESURFER_HOME=/path/to/official/freesurfer/8.2.0-1
FS_LICENSE=/path/to/your/legal/license.txt
FRESH_OUTPUT_DIR=/path/to/new/empty/official_CON01
REPORT_DIR=/path/to/new/report
SOURCE_COMMIT=3b697657919347281d02285b3cd634008635a4c0
PYTHONPATH="$FNIT_SOURCE_DIR/src" CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$FNIT_PYTHON" -u run_backend.py \
  --t1 "$RAW_T1" --freesurfer-home "$FREESURFER_HOME" \
  --license-path "$FS_LICENSE" --output "$FRESH_OUTPUT_DIR" \
  --report-dir "$REPORT_DIR" --source-commit "$SOURCE_COMMIT"
```

通过条件为：首次返回 `completed`，官方 done marker存在且 volume/surface格式及几何检查通过；第二次返回 `skipped` 且同一 subject 目录所有文件内容 SHA-256 保持不变。失败只记录失败状态，不把文件存在视为完成。报告分私有完整记录与公开摘要；公开摘要只含源码、CPU耗时、来源阶段、格式/几何和文件未改变状态。

## 4. 服务器登记

统一根目录 `FNIT` 下：

- 冻结重试源码：`workspaces/connectome_paired_20261003_v1/task04_generated_official/official_retry_v2`。
- 首次失败：`runs/connectome_paired_20261003_v1/task04_generated_official`。
- 新重试：该目录下 `official_retry_v2`；全新输出为其 `fresh_CON01`。
- 持久日志：`logs/connectome_paired_20261003_v1/task04_generated_official/official_retry_v2/run.log`。

实际计时与验收：

| 检查 | 实际结果 |
| --- | --- |
| 同一 API 首次新目录调用 | `completed`；6069.7609066秒，包含完整官方重建和后验检查。 |
| 同一 API 第二次调用 | `skipped`；1.0224293秒，复用同一 subject 目录。 |
| 官方完成与解剖检查 | done marker存在；volume/surface格式与兼容几何通过；anatomy检查记录245个文件。 |
| 全 subject只读复用 | 第二次调用前后349个文件内容SHA逐项相同。 |
| 资源预算 | 官方8.2.0，`-all -parallel -openmp 4`，4个实际CPU亲和性，无GPU。 |

全subject哈希快照是额外的只读验收，未加入两次 API 计时。首次完整耗时没有使用历史记录替代。

## 5. 完成后的同源官方结果比较

[compare_official.py](compare_official.py) 使用项目已有的 `tools/reference/compare_freesurfer_recon_outputs.py`，只读取两份完成的官方 subject。它在本次真实新目录重建成功、第二次调用跳过且 subject 全部文件不变之后执行，CPU计时7.55368秒。实际摘要见 [comparison_complete.public.json](comparison_complete.public.json)。读取的七个科学文件必须匹配各自完成记录中的SHA，比较期间再次核对文件未变。

私有记录先验证两次官方运行的原始 T1 内容相同。比较读取 `brain.mgz`、`aparc+aseg.mgz`、`ribbon.mgz` 和双半球 `white`、`pial`：先检查 volume 网格，再计算体素误差和 label XOR；只有顶点数与有序面索引一致时才计算对应顶点距离。全过程不重采样，不启动 GPU，不修改任一 subject。公开摘要只保存数值，完整影像哈希、文件路径及同源证明保留在服务器。

既有 CON01 官方运行报告记录为 FreeSurfer 8.2.0、8 个线程；本次采用同一安装版本、4 个实际 CPU 亲和性，并启用 `-parallel -openmp 4`。两者实际版本均为 `freesurfer-linux-centos7_x86_64-8.2.0-20260314-d932c45`。既有运行未启用 `-parallel`，CPU亲和性未知。因此本次检验来源适配器的真实调用结果，耗时不能作为同预算的速度比较。

| 科学输出 | 实际一致性 |
| --- | --- |
| `brain.mgz` | 256³；体素值和标量bits改变均为0；RMSE/max误差0；scanner RAS affine、surface RAS vox2ras和网格完全相同。 |
| `aparc+aseg.mgz` | 同上；label XOR=0，前景mask XOR=0。 |
| `ribbon.mgz` | 同上；label XOR=0，前景mask XOR=0。 |
| `lh.white` / `lh.pial` | 各141688顶点、283372面；有序面索引和float32坐标payload完全相同；对应顶点平均/P95/max距离均0 mm。 |
| `rh.white` / `rh.pial` | 各148105顶点、296206面；有序面索引和float32坐标payload完全相同；对应顶点平均/P95/max距离均0 mm。 |

七个输出的科学数组及必要几何元数据全部严格一致。表面文件的创建注释和输出文件名不作为科学一致性条件。该单例结果证明本次官方来源适配器与已有同源官方结果一致；FNIT自有 recon-all 数值等价和十人SC benchmark另行报告。

```bash
# 以下报告包含服务器私有输入身份，只在服务器读取。
FRESH_PRIVATE_REPORT=/path/to/official_retry_v2/report.private.json
TEN_CASE_BINDINGS=/path/to/input_bindings_v1.json
REFERENCE_COMPARISON_HELPER=/path/to/fnit/tools/reference/compare_freesurfer_recon_outputs.py
COMPARISON_OUTPUT=/path/to/new/comparison
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$FNIT_PYTHON" compare_official.py \
  --new-report "$FRESH_PRIVATE_REPORT" --bindings "$TEN_CASE_BINDINGS" \
  --case-id sub-CON01 --reference-helper "$REFERENCE_COMPARISON_HELPER" \
  --output "$COMPARISON_OUTPUT"
```

## 6. 更新记录

- `8bc337c4`：首次真实新目录调用，发现官方完整环境缺失，失败记录保留。
- `3b697657`：完整官方 setup 与 SHA 身份修复；相关CPU测试55项通过；冻结重试完成，第二次跳过且349文件不变。
- `1dd04c0c`、`a4873482`：增加只读CPU同源对照，复用成熟比较函数，核对两份官方完成记录；本例七个科学输出严格一致。

## 7. 代码与参考

- [来源功能说明](../../../docs/connectome/recon_backends.md)。
- [既有科学数组比较工具](../../../tools/reference/compare_freesurfer_recon_outputs.py)。
- [FreeSurfer recon-all 官方说明](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)。
- [FreeSurfer 源码](https://github.com/freesurfer/freesurfer)。
