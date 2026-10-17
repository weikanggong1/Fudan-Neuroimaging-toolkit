# 早期 smoothwm 曲率重复计算清理

## 功能与边界

原生表面配方保留 `white.preaparc` 和 `inflated` 的 H/K，省去随后被
完整 `mris_curvature_stats` 覆盖的早期 smoothwm H/K/K1/K2。Python
放置配方保持原行为。主曲率拟合、最终离散八图、统计定义、TF32 和
网格质量门均保持原算法。没有新增依赖，继续使用主页 Conda 环境。

功能参数、输入输出、具名 Python 示例及原命令见
[曲率说明](../../../../docs/recon_all/CURVATURE_OUTPUTS.md)。本目录工具只做
冻结同输入阶段测试，不参与生产输入链，不把阶段收益称为整例提速。

## 同输入工具和清单

[`benchmark_early_curvature.py`](benchmark_early_curvature.py) 使用两份已完成
的 FNIT 生产 `benchmark.json`，读取各自同级 `subject/`。要求 benchmark
执行完成、退出 0，且 `fnit-native-free-run.json` 的状态也是 complete。
每侧只复制五个自产输入到每次新建的被试目录：

| 输入 | 结构、空间和单位 |
| --- | --- |
| `H.white.preaparc`、`H.inflated`、`H.smoothwm` | 有序三角表面；`(N,3)` 坐标、`(F,3)` 整数面；surface RAS/mm |
| `H.curv`、`H.sulc` | 同顶点顺序 float32 `(N,)` morph；分别为曲率与深度 |

prepare 输出私有 JSON 清单，记录两例四侧、生产 benchmark/runtime、
原始 T1 标识、五输入 SHA-256、源代码提交及完整 `src/fnit/**/*.py`
SHA-256，以及独立构建的 `mris_curvature_stats` 路径和 SHA-256。
清单由工具生成；run 必须显式传入该清单的 SHA，逐项核对后才计算。
源码归档没有 `.git` 时，提交由协调者声明，实际代码由完整 SHA 清单绑定。
程序、输入和源码在完成后再次检查；已有输出目录直接拒绝，失败保留现场。

### 命令行

以下变量均由协调者绑定到已核对的私有路径；许可证从已授权环境继承，
工具不读取或记录其内容。

```bash
curvature_prepare_arguments=(
  prepare
  --benchmark "$benchmark_case_06" "$benchmark_case_07" # 两例成功生产报告
  --source "$frozen_source_root"                       # 冻结候选源码根目录
  --code-version "$frozen_source_commit"               # 完整四十位实际提交
  --native-program "$built_curvature_program"          # Conda 独立构建程序
  --assets "$declared_assets_directory"                # 原生程序声明的资产根
  --output "$curvature_input_manifest"                 # 不存在的新清单文件
)
python validation/recon_all/optimizations/20261017_curvature_schedule/benchmark_early_curvature.py \
  "${curvature_prepare_arguments[@]}"

curvature_run_arguments=(
  run
  --manifest "$curvature_input_manifest"               # 上一步生成的输入清单
  --manifest-sha256 "$curvature_input_manifest_sha256"  # 上一步打印的 SHA-256
  --output "$new_curvature_run_directory"               # 不存在的新运行目录
  --device "cuda:0"                                     # 明确逻辑目标 GPU
)
python validation/recon_all/optimizations/20261017_curvature_schedule/benchmark_early_curvature.py \
  "${curvature_run_arguments[@]}"
```

两例、每侧顺序执行 A/B/B/A：A 使用兼容模式的三表面早期拟合，B 只拟合
preaparc/inflated。两者均调用同一候选函数与独立构建程序，接相同完整
curv.stats 命令；没有把早期连续拟合缓存用于离散统计。每次使用新目录，
完整比较最终八图和另两表面 H/K 的 SHA/字节；morph 另报不同元素数、
最大/P99/平均误差。stats 的原始字节与所有注释头差异单独记录；若头中
含命令路径或时间，不把它当作数值变化。全部非空、非注释统计行均按
原顺序核对非数值结构及所有数值，要求结构一致、有限值精确相等，
不删除任何非注释统计行。上述比较定义在测试前写入报告，不调整容差。
任一 morph 字节/数值变化或 stats 数值/结构变化均报告 comparison_failed。

GPU 阶段前后同步明确设备，固定四核 CPU 亲和性及 Torch/Numba/原生
库四线程预算，不并行执行四侧。记录早期含读写秒数、原生统计秒数、
连续 API 总秒数、复制成本与完整 trial 墙钟；报告首次初始化成本，
不将重叠字段相加。API 时间含实际影像读取、计算、搬运和写出；输入
校验、私有复制、误差比较及采样启动另包含在 trial/外层完整墙钟。

显存每 0.5 秒采样目标 GPU 父子进程同期占用，记录间隔、失败采样及
归属限制；同时记录明确启用分配缓存后的 PyTorch allocated/reserved。
它们不包括所有非 Torch 分配，也不将整卡其他进程计为 FNIT 自身。
共享环境无法确认 PID 对应时自身峰值保留 null。

每次 trial 在原有设备同步之后、peak reset 之前记录入口
`allocated_entry_bytes`/`reserved_entry_bytes`，结束同步后记录
`allocated_current_bytes`/`reserved_current_bytes` 及两个 peak。四次 trial
在同一进程中运行，未清空缓存；`reset_peak_memory_stats` 只重置统计，
因此 reserved 峰值和进程采样可能包含此前 trial 保留的缓存。入口基线
也属于本次 peak 的范围，不能将这些值解释为各模式独立冷进程峰值，
也不能用相减结果代替真实同时占用。若需判断独立峰值差异，另行进行
冷进程显存回归。本次记录不改变 API 计时、A/B/B/A 顺序或算法。

## 本次验证状态

| 范围 | 当前状态 |
| --- | --- |
| 生产修改 | 已实现，候选尚待协调者整例验收 |
| 本地语法、参数帮助与空白检查 | 通过 |
| 本地 native stats 函数隔离测试 | 3 项通过；不代表完整包测试 |
| 完整 8 项调度测试 | 待科学计算环境运行；本地缺少 NumPy/nibabel/Torch/Numba |
| 两例四侧真实 ABBA | 未运行 |
| 两例原始 T1 空目录整例及误差/显存 | 未运行，由协调者统一执行 |
| 整体指标等效 | 未判定 |

测试文件：[test_principal_curvature_schedule.py](../../../../tests/recon_all/test_principal_curvature_schedule.py)。
生产结果与耗时以随后绑定实际提交/程序的机器报告为准。

## 原命令、版本和参考

固定完整统计命令：

```bash
mris_curvature_stats -m --writeCurvatureFiles -G \
  -o ../stats/lh.curv.stats -F smoothwm sub01 lh curv sulc
```

早期二次面主曲率是内部计算，没有独立原软件 CLI。历史同输入完整
八图验证保留在[完整统计说明](../../../../docs/recon_all/FULL_CURVATURE_STATS_PYTHON_20261010.md)，
不能作为本次性能结果。2026-10-17 本次仅清理重复早期写出，不改变此前
生效的曲率和脑区统计计算。

- [固定 FreeSurfer 源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
- Fischl B. FreeSurfer. *NeuroImage*. 2012;62:774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
