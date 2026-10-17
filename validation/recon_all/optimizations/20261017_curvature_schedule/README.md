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
| 完整 8 项调度测试 | 通过，包含在冻结提交的统一 134 项测试中；统一测试 EXIT 0 |
| 两例四侧真实 ABBA | 已完成，原严格比较保持 `comparison_failed` / EXIT 2，原因见下 |
| 两例原始 T1 空目录整例及误差/显存 | 待协调者统一验收；本阶段报告不提供整例结果 |
| 整体指标等效 | 未判定 |

测试文件：[test_principal_curvature_schedule.py](../../../../tests/recon_all/test_principal_curvature_schedule.py)。
生产结果与耗时以随后绑定实际提交/程序的机器报告为准。

### 2026-10-17：曲率同输入 ABBA 与重复性

实际计算源码为 `71cc91899bdf9713690cddbe32054386d8f58d63`。
两种模式均调用该冻结源码：A 保留三表面早期拟合，B 延迟 smoothwm
四图。下表为 A/B/B/A 各模式两次连续 API 的中位数，包含早期曲率、
同一自编译程序的完整曲率统计及读写、搬运和同步；不含私有输入复制。
首次惰性初始化成本仍在观察值内，各 trial 共用进程和分配缓存。

| 公开 T1 | 半球 | A：兼容模式 / 秒 | B：延迟模式 / 秒 |
| --- | --- | ---: | ---: |
| ds000114 sub-06 | LH | 3.3505 | 2.6613 |
| ds000114 sub-06 | RH | 3.1178 | 2.7380 |
| ds000114 sub-07 | LH | 2.7465 | 2.4061 |
| ds000114 sub-07 | RH | 2.6944 | 2.3592 |

最终 smoothwm 八图的 96 次非自比较均字节及解码精确一致，完整 stats
的 12 次非自比较也均字节、全部非注释行结构和数值精确一致。
保留的 preaparc/inflated H/K 严格比较失败，原 `comparison_failed`
和 EXIT 2 保留；没有放宽精度门。相同 A 模式的首次与末次也存在差异。

固定 sub-06 LH 的两表面后另做四次原法线重复计算，并将同一法线固定
后重复四次拟合。首个不同的计算结果出现在已有 `_normals` 的 CUDA
`index_add_` 原子累加：法线最大差为 1.7881×10⁻⁷，P99 为
5.9605×10⁻⁸。原有动态法线拟合的重复最大差可达 3.8719×10⁻⁴；固定
同一法线后，两表面的四次拟合均精确一致。这将本次观察定位到原有
GPU 法线累加重复性，未改成 CPU、未更改生产确定性或精度策略。

生产调用图及实际所选原生 CLI 源码审查未发现上述早期四图的下游重建
消费者：注册、GCSA 和脑区统计从几何计算特征，最终曲率统计独立生成
离散八图。该结论属于静态调用/所选命令审查及上述阶段实测，尚非动态
文件访问审计，也不表示整条 pipeline 的新增误差为零。
详见小型[重复性摘要](repeatability.public.json)；原大报告继续保留私有。

### 其余三侧注册回归：三种比较分别报告

下表比较基线 `5ca07f999375d02281ab60589cb2840e7b488790` 与候选
`71cc91899bdf9713690cddbe32054386d8f58d63`。各侧只做一次新缓存 A/B，
均为同输入、同目标 GPU、四线程、float32/TF32；包含完整阶段读写、
JIT、CUDA 初始化及同步。它们是单次阶段观察，不是整例提速。

| 公开 T1 | 半球 | 基线 / 秒 | 候选 / 秒 | 同预算 A/B 严格比较 | 对历史 FNIT 保存输出 |
| --- | --- | ---: | ---: | --- | --- |
| ds000114 sub-06 | RH | 165.218 | 140.557 | 几何、footer、已保存轨迹精确一致 | 文件 SHA、几何精确一致 |
| ds000114 sub-07 | LH | 112.933 | 103.715 | 几何、footer、已保存轨迹精确一致 | 文件 SHA、几何精确一致 |
| ds000114 sub-07 | RH | 122.178 | 111.618 | 几何、footer、已保存轨迹精确一致 | 文件 SHA、几何精确一致 |

历史整例每侧 worker 为两线程，本轮阶段为四线程。历史已保存轨迹均仅
`rigid_score` 一项不同，绝对差依次为 1.4195×10⁻⁵、2.8836×10⁻⁷、
3.1101×10⁻⁵，其余已保存字段一致。跨预算分数差异单列，不声称线程数
是唯一已证原因；历史完整轨迹严格复现仍未通过。因此 helper 原状态
`completed_with_historical_differences` / EXIT 3 保留，而同预算 A/B
回归通过。所比较的历史保存结果来自 FNIT，不能称为官方重复性验证。
三项比较、提交与报告 SHA 见[三侧注册摘要](remaining_registration.public.json)。
两例端到端误差、完整耗时和同时显存仍由协调者统一验收。

### 共享平均器的 inflation 覆盖

`RegistrationGradientAverager` 也由标准 inflation 复用。参数化工具
[benchmark_inflate_shared_average.py](benchmark_inflate_shared_average.py)
复用已有完整 `run_standard_inflate`、源码清单校验和 inflation 比较器；
没有另写积分器。准备清单已核对两例四侧的实际生产配方：同序 smoothwm
输入、Torch CUDA 后端、`save_sulc=True`、`profile=False`、每侧两线程，
surface worker 的 CUDA 分配缓存开启。父进程的缓存关闭不代表该 worker
也关闭。输出为新空目录中的 inflated、sulc、机器报告和首侧 A/B 只读轨迹。

```bash
# 两份已成功完成的 FNIT 生产 benchmark；不会读取官方检查点。
python benchmark_inflate_shared_average.py prepare \
  --benchmark "${first_completed_benchmark}" \
  --benchmark "${second_completed_benchmark}" \
  --output "${new_inputs_manifest}"

# 两份实际冻结源码、完整 Python SHA 清单及其 SHA；输出目录必须不存在。
python benchmark_inflate_shared_average.py run \
  --baseline-source "${baseline_source}" \
  --baseline-manifest "${baseline_source_manifest}" \
  --baseline-manifest-sha256 "${baseline_manifest_sha256}" \
  --candidate-source "${candidate_source}" \
  --candidate-manifest "${candidate_source_manifest}" \
  --candidate-manifest-sha256 "${candidate_manifest_sha256}" \
  --inputs-manifest "${new_inputs_manifest}" \
  --inputs-manifest-sha256 "${inputs_manifest_sha256}" \
  --output "${new_output_directory}" \
  --device "cuda:0" \
  --python-interpreter "${declared_python_interpreter}"
```

`prepare` 绑定生产报告、实际 API、smoothwm 及历史自产 inflated/sulc 的
SHA；全部清单输入为 JSON。`run` 的 source 为仓库根目录，manifest 为
已有完整源码清单，SHA 参数为其 64 位 SHA-256；`device` 为已由调度器
映射到目标卡的显式 CUDA 设备，`python-interpreter` 为声明的环境解释器。
调度器另持有 GPU 锁并核对实际卡身份。所有坐标为 surface RAS 毫米，
sulc 为同顶点顺序的一维 float32 毫米值。

首侧新 A/B 同时记录完整 60 步，只读观察会同步 GPU，因此其墙钟标为
`diagnostic_not_performance_sample`；另三侧直接复用实际 API。历史 runtime
只有步数、末次 RMS 和停止状态，历史比较仅覆盖这些已保存字段及输出，
不能称为历史逐步轨迹复现。结构检查复用闭合流形、方向和零面积门，
未执行一般三角面碰撞扫描。失败保留非零退出及报告；差异不会改成通过。
准备完成不等于已运行，实际结果须由协调者调度后另行记录。

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
