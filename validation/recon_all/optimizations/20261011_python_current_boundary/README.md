# 当前独立 Python 配方的原版程序边界

## 1. 功能与本次结论

本页保留首版调用链审查，并补充 `0fdf8da6211a` 的最新 Git 源码核对，不运行新的 MRI、GA 或整例。[最新静态报告](reports/static_0fdf.json)覆盖60个源文件，[绑定核验](reports/static_0fdf_binding.json)逐项确认Git blob的大小、SHA和函数行范围。它没有发现显式 Python＋GA 配方可达的原版阶段程序或失败回退；公开缺省仍为混合 native，整例和隔离部署均未在此审查中验证。下表及原 `audit.json` 保留首版的行号与证据版本。

首版代码基线是 `748b86208c5c4e18075321cd13005cdc1fc8e53a` 的完整 `src`，仅覆盖 root 捕获的 `native_free.py` 缓存开启接线；该文件 SHA 为 `f9587540dc97225287da7cb24f52e8b297e6a69aa04faf5101ed5f04a0bf2021`。原工作树中他人的 retessellation 改动不进入快照：审查版本为 Git 中的 `4b20a92f…`，不是工作树 `89d7423f…`。

本次重读 24 组调用链、90 个函数引用和 62 个文件，并核验完整快照的 1370 个文件。`backend="python"` 的默认数字拓扑与显式 `topology_backend="python-ga"` 均没有选中原版 FS 阶段程序、原版链接库或原版编译回退。此结论是带 SHA 的手工分支审查；当前整例执行、共享库加载和 OS 隔离没有在本任务中验证。

```mermaid
flowchart LR
    A[公开入口与21项配方] --> B[程序发现条件]
    B --> C[N4和体积链]
    C --> D[数字拓扑或显式完整自有GA]
    D --> E[表面、曲率、缺陷投射与统计]
    E --> F[MNI和网格检查join]
    F --> G[138项文件存在检查]
```

没有发现新的原程序泄漏接口错误。下一优先项是验收现有完整 GA 文件适配器和生产链，而不是再实现同名模块；WM 活跃路径分支的真实数据覆盖仍缺，N4 已测的强度及下游差异也仍需保留。

## 2. Python 调用、输入与输出

审计函数只有三个必填 `Path` 参数：

| 参数 | 输入内容 | 行为 |
| --- | --- | --- |
| `repository` | FNIT 仓库根目录 | 提供已提交的旧审计函数、24组结构、历史公开收据和 Git 对象；不作为当前生产源码导入路径 |
| `source` | 独立源码快照目录 | 含完整 `src` 和 `SOURCE_MANIFEST.json`；逐文件校验大小/SHA，并重新读取声明的 Git archive，拒绝未声明改动 |
| `output` | 尚不存在的目录 | 写出本次审计；目录已存在则失败，避免覆盖旧结果 |

`SOURCE_MANIFEST.json` 必须含 `git_commit`、`archive_sha256`、`archive_bytes`、`files` 和 `overlays`。`files` 逐项含 `sha256`、`bytes`；显式覆盖项另含 `git_sha256`、`captured_sha256`、`bytes`。清单使用仓库相对路径，不接受绝对路径或 `..`。本次唯一覆盖项为上述 `native_free.py`。

```python
import importlib.util
from pathlib import Path

repository = Path("/work/FNIT/repo")  # 通用示例目录，不是服务器地址
source_directory = Path("/work/FNIT/audit-source")  # 已核验的Git源码及显式覆盖清单
output_directory = Path("/work/FNIT/audit-result")  # 必须是新目录
audit_script = repository / "validation/recon_all/optimizations/20261011_python_current_boundary/audit_current.py"
module_spec = importlib.util.spec_from_file_location("current_audit", audit_script)
audit_module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(audit_module)
result = audit_module.collect(
    repository=repository,
    source=source_directory,
    output=output_directory,
)
print(result)  # 组数、函数引用数、历史事件数和本次report SHA；不是MRI结果
```

输出 `audit.json` 记录 24 组、逐函数 SHA/行号、13 组原程序条件、三种显式配方、缓存/资产边界、原始与公开历史 SHA 及验收缺口；`reviewed_source.json` 记录本次被审文件的 SHA/大小；`migration_table.md` 提供简表。无体积、坐标数组或参考预测进入输出。函数定位与历史执行收据核验复用既有 `20261011_python_call_chain_audit/audit_call_chain.py`，旧表只复用结构，不复用旧结论。

## 3. 命令行调用

本工具用 Python 3.11 标准库运行，无需导入 PyTorch、初始化 CUDA 或安装新依赖：

```bash
repository=/work/FNIT/repo
source_directory=/work/FNIT/audit-source
output_directory=/work/FNIT/audit-result
python "$repository/validation/recon_all/optimizations/20261011_python_current_boundary/audit_current.py" \
  --repository "$repository" \
  --source "$source_directory" \
  --output "$output_directory"
```

当前被审生产入口是 `fnit-recon-all` → `native_free.main` → `run_recon_all_python`；批处理调用同一个 `python_profile_options`，没有另一套默认。Python 配方保留显式 Torch N4 和 `python-ga`，签名中历史 `native` 缺省分别解析为 SimpleITK 与数字拓扑。4 线程双侧 worker 各 2 线程，worker 是新的 Python exec。

## 4. 原软件调用与实际选择

本任务不执行原软件。完整条件表见 [migration_table.md](reports/migration_table.md)，每项入口/行号/SHA见 [audit.json](reports/audit.json)。

| 当前入口 | 独立 Python 实际路径 | native 对照分支 |
| --- | --- | --- |
| `input_n4_chain.py:36` | 默认完整 SimpleITK N4，或显式完整 Torch N4；拟合默认1线程 | `fnit_n4_itk` |
| `native_free.py:78` | 有序 Numba WM分割、Torch直方图/平面，独立Python worker | `mri_segment` |
| `native_free.py:138` | 完整有序 CPU WM/aseg 编辑，包含 typed/byte 路径搜索 | `mri_edit_wm_with_aseg` |
| `native_free.py:259` | 显式自有 GA 文件接口，独立目录写出并校验报告、三输出 SHA、最终网格门，再发布 | `mris_fix_topology_fnit` |
| `native_free.py:609` | 自有离散曲率、Numba 统计；`source_libm` 是 `np.arccos` 等通用数学规则 | `mris_curvature_stats` |
| `native_free.py:639` | Torch 投射原域缺陷标签，左→右更新共享 `surface.defects.mgz` | `mri_label2vol` |
| `native_free.py:1057` | 自有 SurfaceStatsCache、Python/Numba ROI行与全局汇总 | 无原统计程序调用 |

native 缺省 CUDA 链仍执行 N4、WM分割/编辑、拓扑、inflation、white/pial、atlas paint、曲率统计和缺陷投射的原程序。默认 CUDA 的 GCA `auto` 已走自有 Torch；原 `mri_em_register` 是 `original`/CPU对照路径。CUDA 表面指标和 MNI postprocess 已自有；MNI 三个原程序仅在非 CUDA 对照分支。不能把发现 `mris_place_surface` 供原 white/pial 使用误计为 CUDA 指标也运行了原程序。

公开入口缺省仍是 `backend="native"`，没有在本轮切换默认。脱离原程序需明确 `backend="python"`；当前配方还要求显式 `cuda:N`，因为成熟 MNI/WM 路径仍需 GPU。有序 CPU/Numba 子步骤允许保留，这不等于 `python-gpu` 全GPU配方已经完成。

资产运行预检 `assets.py:400` 只核显式目录中固定文件的大小/SHA；没有调用安装器。安装器 `assets.py:261` 可复制已核验的 distribution **数据资产**或按许可从 Release/作者下载，这不是运行/编译原阶段程序。外部权重和模板仍各自遵守许可。独立 Python 路径不读取 FS 许可证；它也不依赖原程序目录是否在 `PATH`。

球面配准 CUDA 平均绕过可选 CPU C++ 平均器；SynthSeg CUDA 绕过可选 columns C++；SynthMorph CUDA 绕过 CPU Eigen glue。三条未选中路径另有函数 SHA。SimpleITK/ITK、Torch/CUDA/Triton、NumPy/SciPy、Numba/system libm 是允许的通用库。缓存未新增依赖或 FP16/BF16：面缓存每缺陷固定基底，动态邻接/新增面按候选重新消费；state cache 在本路由默认未启用。

## 5. 实测证据与剩余验收

本任务没有新精度/时间/显存 benchmark。机器报告读取并绑定已有原文件 SHA，保留其 producer 与当时的作用范围：

| 已有真实证据 | 实际结果 | 本轮不能据此宣称 |
| --- | --- | --- |
| 完整 SimpleITK 文件API，sub06/sub07 | 对原程序量化图分别99/28体素相差1，P99=0；几何一致 | 与原程序逐位相同，或这些变化无下游影响 |
| 完整 WM 分割，两例阈值修复版 | 同输入原/自有结果0体素差 | 当前新生产整例通过 |
| 完整 WM编辑 CPU/hybrid 两例 | 解码0差、前景Dice1；原文件bytes不同 | raw文件逐字节相同 |
| 十例 WM检查点 ROI扫描 | 未找到活跃路径分支 | 活跃分支已获真实原生成功对照 |
| 新显式面缓存 core False/True阶段 | 5次初始+1次final评分、81字段/4NPZ/控制/RNG一致 | 当前缓存开启文件适配器、全部GA或整例通过 |

阶段脑图可查 [N4 sub06](../20261010_n4_simpleitk/plots/sub06_n4.png) 和 [WM sub06](../20261011_wm_remove_paths/plots/sub06_wm_paths.png)。这些是已授权公开数据的旧阶段衍生图，未改标为本轮结果。

旧冻结 v4 的实际启动审计，sub06为839事件、sub07为10940事件，事件记录包含 Python/通用编译辅助程序。原 blocked-stage 清单为空；两例仍为质量失败，执行只覆盖失败之前。其 producer、原 JSONL SHA、导出原 SHA 和公开 SHA 保存在机器报告，不能作为 `748b+f958` 的执行次数证明。当前原程序尝试数、当前共享库尝试数都为 `null`，不是实测0。

下一步具体门：先完成当前完整 GA 的全部缺陷、remesh3、严格 ORIGINAL 清理和质量门，再冻结当前完整生产链记录；出现质量首差后按实际坐标/面/持久状态定位。WM真实 active 范围继续保留未测；无法定义安全原存储/越界或不终止时明确失败。N4保留完整后端和已测连续误差，不通过更换默认掩盖差异。138项检查只是 join后的文件存在检查，网格质量/数值一致/整体等效是分开的结论。

## 6. 版本与审查记录

- 旧 `20261010_python_profile_static`：固定旧 v4 的24组结构，数字拓扑，不是当前 GA 接线。
- 旧 `20261011_python_call_chain_audit`：当时24组76引用；当前工具复用其逐函数 SHA 与历史事件核验。
- 本次 `748b+f958`：24组90引用、13组原程序条件、21项配方；显式完整 GA 已接入，root 路由开启面缓存，adapter/core 缺省仍False。1370个 Git/覆盖文件全部核验。
- `0fdf8da6211a` 补审：60个实际Git源文件及482个函数行范围完成绑定核验；显式Python/GA的原程序分支、21项配方、CUDA目标、线程预算、精度例外和worker失败不回退均重新查看。482是源内函数定位数量，不代表482项算法测试。未消费未提交的距离接线或pial残余修改，也未执行新MRI/GPU。
- 当前没有新整例/MRI、动态链接器或 OS 隔离结果；整体等效 `not_assessed`，新增原程序泄漏问题未发现。代码/依赖版本由声明快照绑定；不把本机安装包版本冒称 accelerator-b 环境。Python>=3.10、Torch>=2.1、NumPy1.x、Numba>=0.59、SimpleITK==2.5.6等声明及主页 Conda 的 Python3.11/Torch2.5.1/Triton3.1.0见机器报告；缓存没有新依赖。

## 7. 参考文献与源码

- 本仓库逐函数源码、当前条件表和历史收据：[audit.json](reports/audit.json)、[reviewed_source.json](reports/reviewed_source.json)。
- 原算法与独立对照程序来自固定 FreeSurfer `d932c45` 源码构建；[FreeSurfer 仓库](https://github.com/freesurfer/freesurfer)。本页不复制发布原 C++ 源码或原程序。
- Tustison NJ et al. N4ITK: Improved N3 Bias Correction. IEEE TMI, 2010；当前默认通用 [SimpleITK N4 API](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html)。
- Fischl B. FreeSurfer. NeuroImage, 2012；原算法参考与现有模块的独立实现、阶段验证和整例验收分别记录。
