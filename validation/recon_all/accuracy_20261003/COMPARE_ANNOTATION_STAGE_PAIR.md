# Annotation stage 同输入配对比较

## 功能、输入与输出

`compare_annotation_stage_pair.py` 是纯 CPU、只读的比较工具，输入两个 `replay_annotation_stage.py` 产生的 `annotation.stage.json` 和全新的输出目录。不导入 PyTorch，不启动 GPU，不运行 annotation 算法，不读取官方结果。依赖仅 NumPy、Nibabel，复用既有 FNIT Conda 环境。

先要求两个 stage_complete、6 个 lh/rh×aparc/aparc.a2009s/aparc.DKTatlas 输出、7 个 subject 输入和8个资产的 `(kind,relative,size,SHA)` 绑定一致，以及 stage 内前后绑定稳定。要求父进程实际 TF32=true、autocast=false，worker 成功并记录 TF32=true，实际 Torch/Numba 有效线程等于该组预算；两臂的线程配置和精度一致。新旧源码允许不同并分别报告来源，运行时/解释器/源文件SHA及分步时间不改标。

每个 `.annot` 和对应语义 NPZ 先核对 file SHA/size，NPZ 用 `allow_pickle=False`，再核对各数组回执 SHA，独立以 Nibabel 读原 `.annot` 验证 NPZ 对应原文件。四个 smoothwm/sphere.reg 原文件也核对 SHA，再读真实坐标与有序 faces 检查数组 SHA。只有两臂的坐标和有序面完全相同才按 vertex index 比较标签；拓扑或坐标不同则跳过逐顶点比较，exact pass=false。

逐 hemi/atlas 分别比较以下数组，报告不一致数量及全部差异位置和值：

- `original_annotation_ids`：原始 packed RGB ID，**不是解剖类别编号**。
- `label_table_indices`：Nibabel orig_ids=False 的 color-table 行索引，未标注可能为 -1。
- `vertex_indices`：0..nvertices-1 的有序记录索引。
- `color_table`：包括 RGB、透明度和 packed ID 的五列整数表。
- `names`：原始名称 bytes 按十六进制报告，避免文本解码混淆。

Dice 使用原始 packed ID 在相同 vertex 上的集合；逐标签报告两臂数量、交集、Dice、对应基线 color-table 行/名称。min/p05/median 汇总两臂实际出现标签的并集（双方均未出现的标签不纳入汇总）。color-table 或名称不同仍记录数字集合的 Dice，但 `dice_named_semantics_comparable=false`，不能解释为相同解剖区域的 Dice。原 annot/NPZ 文件SHA不同单列为字节差异，不当作 label 变化。

输出 `annotation.pair.json`、简短 `README.md`，包含原始文件SHA、source provenance、两臂 runtime、timings、准入条件、四mesh比较、六annotation数组差异和Dice。`exact_regression_pass` 仅当所有前提与 labels/table/names/coords/orderedfaces 全部一致才为true。`official_equivalence`始终`not_assessed`；不是整例或官方等效验收。

## Python 调用

```python
# CPU-only；报告中的两臂真实运行时间不会被比较脚本时间替代。
from pathlib import Path
from compare_annotation_stage_pair import compare

comparison_report = compare(
    Path('/absolute/baseline/annotation.stage.json'),
    Path('/absolute/candidate/annotation.stage.json'),
)
print(comparison_report['exact_regression_pass'])
```

## 命令行调用与参数

```bash
FNIT_COMPARE_PYTHON=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python
"${FNIT_COMPARE_PYTHON}" compare_annotation_stage_pair.py \
  --baseline-stage /absolute/baseline/annotation.stage.json \
  --candidate-stage /absolute/candidate/annotation.stage.json \
  --output /absolute/new_annotation_pair_report
```

`--baseline-stage`、`--candidate-stage` 必填，分别为已完成基线/候选 stage JSON 的绝对或当前目录相对路径；回执内文件按记录的实际路径读取。`--output` 必填，必须不存在且与两个stage目录分离；只在此目录写报告。没有 GPU、重跑、官方文件或精度切换参数。退出码0为严格通过，1为有效比较有差异，2为输入/回执/结构无效；失败也写 JSON 原因，不用失败结果补生成标签。

## 原软件与验证记录

本工具不对应一个 FreeSurfer recon-all 命令；它比较已产生的 `.annot` 文件，格式读取采用 [Nibabel read_annot](https://nipy.org/nibabel/reference/nibabel.freesurfer.html#nibabel.freesurfer.io.read_annot) 的 orig_ids 两种语义。参考 [FreeSurfer annotation 格式](https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles)；不调用 FreeSurfer。

2026-10-04：已完成脚本语法编译检查。真实两病例×六 annotation 回归由协调者使用既有 stage 结果运行，本文件不预填精度、耗时或脑图结果，不把模拟数据称为 benchmark。可选脑图尚未实现；机器可检查的标签/拓扑/文件绑定比较已提供。
