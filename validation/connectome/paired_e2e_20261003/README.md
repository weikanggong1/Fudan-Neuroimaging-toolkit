# 配对模板入口：独立 CPU 集成验收

## 范围与版本

2026-10-03，在 gpucw1 既有 FNIT Conda 环境中检查公共 Python/CLI、真实模板读写与映射、检查点及矩阵构建。最终冻结源码为 `d75fce8937e7aa4c719c4733960c89eb2e27f0aa`，**97 项通过，16.68 秒**。环境与六个实际生产模块 SHA-256 见 [provenance.public.json](provenance.public.json)，完整输出见 [focused_tests.txt](focused_tests.txt)。

本组使用明确的小型 MRI、FreeSurfer 表面和 annotation 算例验证接口。仅把昂贵的 `_compute_shared_core` 换成三条已知轨迹；该替换包含配准、FOD、追踪和 SIFT2 求解。模板加载、surface→volume、ROI 规范化、端点配对、四矩阵聚合、内容哈希、缓存恢复及 CLI 输出均调用实际生产函数。**这是集成测试，不是十人的影像精度或耗时 benchmark。** 十人真实评测由协调者单列，不能以本报告推断 FNIT recon-all 或追踪与官方数值等价。

## 检查结果

| 场景 | 验收结果 |
| --- | --- |
| 两个 surface 模板 | 6×2 矩形矩阵和完整 ROI 表符合逐项手算结果 |
| 两个 volume 模板 | 2×3 矩形矩阵符合逐项手算结果 |
| surface + volume | 6×3 矩形矩阵符合逐项手算结果 |
| 相同模板 | count、SIFT2 FBC、mean length、mean FA 与既有方阵构建函数逐值相等 |
| A→B→A | 共享数值链仅调用一次；返回 A 时命中 A 的模板和矩阵检查点 |
| 只改端点半径 | 共享链及模板映射复用，仅矩阵重新计算 |
| 同大小、同 mtime 改模板内容 | SHA-256 检出变化，模板/矩阵失效，共享追踪仍复用 |
| 损坏矩阵 NPY | 正确重算矩阵，保留有效共享链及模板映射 |
| CLI 三组模板 | 实际 CLI 生成 27 个声明输出；JSON 资源路径相对 JSON 所在目录解析 |
| 输出 ownership | A、B 历史输出可复用；用户手工修改 CSV 后拒绝覆盖 |
| 坏模板早拒绝 | 分数标签、4D 标签、缺失 ROI 的 TSV、原生表面顶点数不匹配及空模板组均在共享链前报错 |
| 提供 recon-all 只读 | 正常运行前后所有输入文件字节一致；output/checkpoint 与输入重叠或目录 symlink 指向输入时提前拒绝 |
| recon 选项 JSON | 文件内相对路径以文件目录为基准，内联 JSON 以当前目录为基准；裸 `recon-all` 通过 PATH 查找 |
| 任意 MNI 标签网格 | 1 mm ROI 网格与 2 mm 配准参考共用世界 RAS，nearest 采样符合明确期望，未二次重采样标签 |

原有 BIDS/recon 来源 34 项及模板、共享检查点回归一起执行；97 是全部七个测试文件的总数。

## 本次修复

- **早期输入验证**：以前直到共享追踪完成才检查模板的整数标签、LUT 和原生表面顶点数。新增 `preflight_template()` 与 `preflight_template_pairs()`，只读 CPU 验证后再进入 BIDS 预处理及共享数值链；尚未生成 recon-all 时推迟依赖原生网格的检查，生成后立即补查。
- **用户 subject 保护**：新增 `validate_readonly_subject_outputs()`，解析 symlink 后拒绝输出、检查点与用户提供的 subject 相同、包含或被包含，防止输出写入输入。
- **协调者同步修复**：选项支持内联 JSON；官方裸命令通过 PATH 查找；配对缓存校验 dtype/shape/数值依赖，发布前重核对来源；MNI 标签按世界位移采样任意源网格。这些变更分别由协调者实现并纳入本次测试。

这些变更不改变连接定义、追踪参数、SIFT2 或矩阵统计。

## 复现

在装好主页 Conda 环境的独立源码目录运行。线程限制仅用于本次 CPU 测试，不是生产 pipeline 的固定资源参数。

```bash
FNIT_SOURCE_DIR=/path/to/Fudan-Neuroimaging-toolkit
FNIT_PYTHON=/path/to/conda/environment/bin/python
cd "$FNIT_SOURCE_DIR"
PYTHONPATH="$FNIT_SOURCE_DIR/src" CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$FNIT_PYTHON" -m pytest \
  tests/connectome/test_paired_e2e.py \
  tests/connectome/test_paired_mni_grid.py \
  tests/connectome/test_template_pairs.py \
  tests/connectome/test_checkpoints.py \
  tests/connectome/test_pipeline_checkpoints.py \
  tests/connectome/test_bids.py \
  tests/connectome/test_recon_backends.py -q
```

源码和测试产物登记在服务器统一索引的 `connectome_paired_20261003_v1/task04_e2e`。测试不调用官方 MRI 程序，不使用 GPU，没有启动或改写十人 benchmark。
