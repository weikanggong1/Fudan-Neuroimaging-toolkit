# 初始种群前三候选的几何谓词缓存诊断

## 1. 功能简介

本实验检查同一缺陷的前三个真实初始候选能否复用相同有向四顶点的几何判定。复用现有 `EdgePredicateMemo`、`CandidateCycleEvaluator` 和种群控制器；只新增独立诊断工具，不改变生产流程或算法公式。

一条链只准备一次评估器和不可变 canonical，只建立一个容量250000的memo。三个候选依次执行完整评分闭环，保留前一个候选反馈的 rip、use_counts、fitness，以及选择性DVS恢复后仍存在的K/E/fx/curvbak等状态。动态邻接、valid、marks和首次命中仍逐次计算，没有缓存候选输赢。

```mermaid
flowchart LR
  A[自产 nofix / qsphere / brain] --> B[成熟基底及首缺陷准备]
  B --> C[一份 evaluator 与 memo]
  C --> D[初始 slot 0：identity]
  D --> E[初始 slot 1：分组与可能变异]
  E --> F[初始 slot 2：分组与可能变异]
  F --> G[完整状态 / 连续RNG / 控制 / 计时收据]
```

诊断调用 `population_size=3, max_unchanged=0, niters=0`，仅取初始前缀。成熟代码先串行生成初始种群，再进入迭代；这三个参数没有进入前三候选的排序公式或反馈更新。小规模合同使用非恒定反馈，确认其排序、变异、分组顺序、完整随机流与正常十候选配方的前三项一致，也核验 `niters=-1` 在第四回调之前的同一前缀。这里没有执行完整GA或生成最终修复表面。

## 2. Python调用、输入和输出

```python
from benchmark_recon_topology_edge_predicate_population_prefix import run_initial_prefix

prefix_result = run_initial_prefix(
    search_arguments=complete_search_arguments,  # 全部有序E域、分组、overlap及初始rip/RP状态
    draw_random_number=continuous_vnl_stream,  # 当前链seed=1234的同一个VNL实例，不能逐候选重置
    evaluate_patch=prepared_candidate_evaluator,  # 同一完整评估器，连续保留跨候选状态
    normal_population_size=10,  # 用来检查普通配方仍存在前三项；不执行十候选
    normal_max_unchanged=10,  # 普通配方的裁剪参数；诊断停止参数固定为0
)
```

此函数属于仓库诊断工具，需把冻结包的 `tools` 和 `src` 放入Python路径。

| 输入/参数 | 格式、单位和限制 |
|---|---|
| `search_arguments` | 完整具名字典，参数与成熟 `run_population_search` 相同，排除种群/停止/回调参数；没有影像或参考结果的默认兜底 |
| `edge_count` / 分组 | 非负E、原序segment_edge_numbers和segment_rows；overlap的CSR及不完整标记也必须完整提供 |
| 初始状态 | `translated_defect_vertices`为整数(K,)；discarded为bool(K,)；rip为signed-char值的int32(N,)；use_counts为非负int32(K,)；vertex_fitness为有限float32(K,) |
| 删除策略 | `vertex_eliminate`为bool，`deletion_count`为正整数，`elimination_threshold`为非负float32标量；前三初始候选不会进入代间删除 |
| `draw_random_number` | 同一链连续调用 `TopologyRandomStream(seed=1234)`，记录每次序号、low、high、value四字段；不增加或改写源抽样 |
| `evaluate_patch` | 完整评分回调，以具名ordering/discarded/rip/use_counts/fitness输入，返回有限fitness与更新后的完整同域反馈；不逐候选重建评估器 |
| 普通配方参数 | 上述两个默认均10；成熟裁剪后少于3候选时拒绝，不能暗中换输入 |

真实工具读取三项FNIT自产文件：`orig.nofix`和`qsphere.nofix`须有相同有序面；坐标转为原流程float32 surface/canonical RAS，单位mm；`brain.mgz`保留原网格、体素类型和RAS到voxel矩阵。文件路径及SHA由私有config提供，不读取官方、旧候选或已知最佳排序。缓存使用准备器复制的完整bytes-backed不可变float32 `(N,3)`，不同对象/元信息失败；无需新权重或依赖。

返回结构沿用成熟种群控制器：best_ordering/int32(E,)、best_discarded/bool(K,)、best_fitness/F64，population_orderings/int32(3,E)、fitness/F64(3,)、ranks/int32(3,)，最终rip/RP数组、删除参数、evaluation_trace、空generation_trace、8项计数和秒数。`complete_ga=False`、`final_tessellation_complete=False`。非法输入和不完整反馈抛ValueError；真实运行异常保留退出码和日志，不发布完成收据。

每臂从同一自产基底独立执行cold、warm、audit三条链，每条均只准备一次评估器/memo；链间不共享缓存。每链三候选保存36项候选NPZ、40项持久状态NPZ、14项排序/输入反馈NPZ和6项控制JSON；另保存完整结构化RNG流、14项prefix NPZ及7项prefix控制，共45个artifact。完成收据绑定全部SHA，不公开这些私有数组和原始日志。

## 3. 命令行调用

```bash
# 此处0、1仅为本机示例；实际必须与已批准协议/config的亲和性一致且先核资源。
taskset --cpu-list 0,1 env CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 NUMBA_NUM_THREADS=2 \
  PYTHONPATH=/data/frozen_prefix/src:/data/frozen_prefix/tools \
  python /data/frozen_prefix/tools/benchmark_recon_topology_edge_predicate_population_prefix.py \
  --pair /data/private_baseline.json /data/private_memo.json /data/new_prefix_pair
```

`--pair`的三个具名含义依次为baseline配置、memo配置和必须不存在的控制器输出。每臂是新进程、独立空Numba缓存、相同两线程预算；第一臂完成退出后才启动另一臂，双方完整收据到达后才比较。单臂调试使用 `--config`和`--arm`，须另设该臂 `NUMBA_CACHE_DIR`。

配置字段：`inputs`含nofix/qsphere/brain路径；`input_sha256`逐项绑定；`source_root`、`source_manifest`及SHA声明全部载入源码；`protocol`及SHA绑定预设规则；`output`必须为空新目录；hemisphere须与本协议固定的lh一致，启动前拒绝不符值，并在实际报告和比较器再次核验；threads固定2；cpu_affinity为两核列表；maximum_entries固定250000；numba_cache_dir为独立空目录。

临时ordering观察器和scanner factory均记录 `global_function_replacement=True`，不改传参或返回，finally恢复，并在成功后检查实际对象恢复。两臂严格检查完整114组：跨臂和自身重复的全数组shape/dtype/bytes、控制、完整RNG序号/区间/值、排序/变异和累积反馈，以及审计有序谓词bool SHA。共同缺字段、不完整artifact、错误actual path/origin/SHA或重复间缓存未复位都拒绝。

独立 `export_metrics.py`在双方收据后重新执行全部门，核对原比较及冷CLI退出记录，再导出预定义数值/hash。时钟分别保存整臂冷CLI、自产加载/准备、每链完整API及读写、逐候选闭环/扫描/重连与写出、memo释放，CPU RSS按字节记录。嵌套时钟不能相加当CLI，独立冷JIT不能与另臂热JIT混比。

此次原比较器因JSON键序失败，该成功导出器没有产生成功结果。另用 `recheck_prefix_json_order.py`读取同一冻结产物：复用SHA绑定的完整原比较器，只在独立函数副本中把prefix控制JSON的键序比较换为完整对象检查。7项键集合、嵌套类型和列表顺序全部保留；bool不能等于int，int不能等于float，浮点按F64位比较并区分负零，拒绝重复键和非有限值。原函数、源文件、artifact和失败收据均不改，没有重新评分。

```python
from recheck_prefix_json_order import export as export_prefix_recheck
from export_execution_receipt import export as export_prefix_execution_receipt

prefix_metrics = export_prefix_recheck(
    baseline="/data/private_pair/baseline",  # 已完成baseline臂的45个artifact与收据目录
    memo="/data/private_pair/memo",  # 已完成memo臂的同一套artifact；不得补跑替换
    controller="/data/private_pair/controller",  # 原3/114失败比较与两臂退出记录
    source_root="/data/frozen_prefix",  # 实际测试的104文件冻结源码，包含原029入口
    source_manifest="/data/frozen_prefix/SOURCE_MANIFEST.json",  # 全部文件大小/SHA及实际模块映射
    protocol="/data/frozen_prefix/PROTOCOL.json",  # 实际执行的资源协议，科学门保持原样
    output="/data/private_pair/recheck/metrics.public.json",  # 必须不存在的typed输出；不写私有数组
)

execution_receipt = export_prefix_execution_receipt(
    workspace="/data/private_workspace",  # 含frozen与本次外层worker脚本的独立工作区
    run="/data/private_pair",  # 含原EXIT1、STARTED、配置和资源快照的运行目录
    original_freeze="/data/old_prepared_freeze",  # 保留的初版资源协议，仅供核对启动前资源重绑定
    metrics="/data/private_pair/recheck/metrics.public.json",  # 完整离线重核后才允许读取
    output="/data/private_pair/recheck/execution_receipt.public.json",  # 新的typed完整CLI/资源收据
)
```

上述路径参数为文件/目录，无单位；不提供默认路径。`prefix_metrics`为具名JSON对象：`arms`按baseline/memo、cold/warm/audit、slot0/1/2保存完整时钟、RSS字节、缓存累计与差量、随机流SHA和来源SHA。`execution_receipt`保存外层CLI秒数、原退出码1、原111门/3失败与3对象门、资源窗口和冻结文件绑定；不含主机、账号、PID、私有路径或原始数组。源/收据不匹配、缺artifact、其它科学门失败或已有输出时抛异常。独立重核CLI参数为上述相同名称加`--`；示例参数必须指向实际冻结目录，不能改用当前工作树。

## 4. 对应原软件

这是 `mris_fix_topology -ga`内部的初始种群排序和有序边相交判定，没有独立官方CLI。源码语义参考固定FreeSurfer版本的 [mris_fix_topology](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mris_fix_topology) 与 [mrisurf_defect.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)。实验复用FNIT成熟Python/Numba实现，不复制或运行原版C++；这里的baseline是同源码未缓存FNIT，不是新跑的官方结果。

## 5. 最新精度和耗时

本次真实配对已结束，科学基线固定为Git `e9e89cac`、原入口SHA `029a8bc6`、科学源码清单SHA `cba6ef26`，104文件共1249165字节。两臂在同一节点、相同两线程、CPU67–68上顺序冷启动；GPU隐藏。运行前已将初版CPU92–93资源重绑定为67–68，实际协议SHA `17e747ea`；旧协议和未启动的忙碌快照保留，科学输入、cap、seed、公式和验收门均未改。启动前1.014秒窗口未观察到窄亲和冲突或重FNIT占用，宽亲和忙碌为46.08%/43.43%；这是共享环境的一次观察，未保证机器独占。

**严格逐文件复现未通过。** 原114组比较中111组通过，3组prefix控制JSON逐字节不同，外层退出码1，原失败SHA `6fa30d8a`保留。实际原因是原工具用frozenset构造JSON字典，两份冷进程写出的键序不同；cold/warm/audit的三份对象均为完整7字段，各叶类型、值和F64位相同。独立离线重核保留111个原门，另验证3个完整对象门；不把原失败改记为114项逐字节成功。

**同输入初始前三候选未出现状态、控制、排序、累积反馈、随机流或谓词审计退化。** 每链实际10447次随机调用的序号、上下界和值逐位一致，全部36候选/40持久状态/14排序字段和6项候选控制一致。该证据不覆盖完整495候选搜索、完整GA、最终网格或原始T1整例；整体等效未判定。

| 本次实测范围 | baseline | memo | 相对变化 |
|---|---:|---:|---:|
| 每臂冷CLI：导入、校验、FNIT自产准备、9个候选、全部读写、退出 | 162.318 s | 159.541 s | -1.71% |
| cold三候选完整API与输出、释放 | 40.106 s | 40.749 s | +1.60% |
| warm三候选完整API与输出、释放 | 22.958 s | 23.114 s | +0.68% |
| warm三候选闭环合计，不含artifact写出 | 6.362 s | 7.236 s | +13.73% |
| warm扫描API合计 | 0.626 s | 1.291 s | +106.07% |
| warm重连内部时钟合计 | 2.746 s | 3.815 s | +38.91% |
| CPU进程RSS峰值 | 1732001792 B | 1774157824 B | +42156032 B |

两臂自产准备的观测耗时不同，base准备51.098→48.017秒、priors/edge准备15.044→13.477秒；冷CLI下降不能归因于memo提速。memo的warm扫描和闭环均更慢，未证明该缓存有性能收益，不接入生产或提升默认。整对外层CLI为355.465秒，包含双方冷进程、全部输出及原失败比较；另一次离线重核为35.812秒，另计。单臂RSS峰值没有相加为并发峰值，父子进程合计峰值未测；GPU allocated/reserved为null。

| 同一链的slot | 逻辑谓词数 | 缓存命中 | slot命中率 | 超容量重算 |
|---|---:|---:|---:|---:|
| 0 | 351038 | 0 | 0% | 101038 |
| 1 | 371086 | 187737 | 50.59% | 183349 |
| 2 | 343150 | 179125 | 52.20% | 164025 |
| 合计 | 1065274 | 366862 | 34.44% | 448412 |

三个repeat的计数一致；audit两臂的完整有序谓词序列一致。cold/warm仅记录计数，完整序列摘要为null，不据此宣称其审计序列已验证。首slot填满250000项容量；超容量仍执行原谓词，没有截断候选扩展。总unique key数量为null，不能把容量250000写成全部独特key数。完整指标、原失败绑定及离线重核来源见[专项收据](../../validation/recon_all/optimizations/20261011_topology_edge_predicate_population_prefix/README.md)。本实验不生成最终表面，因此没有新整例脑图、表面质量或安装隔离通过结论。

此前 [v5首identity实验](TOPOLOGY_EDGE_PREDICATE_MEMO.md) 已完整交付负结果：351038次逻辑谓词、命中0、250000项缓存，完整冷CLI130.130→131.456秒，warm完整候选API8.434→9.535秒。v5每个repeat重建缓存，没有测试跨候选复用。v6使用同一输入和cap，只扩大为前三初始候选，不通过换阈值、参考或赢家改善数字。当前生产默认仍是成熟未缓存路径。

## 6. 更新和benchmark记录

- 2026-10-11，v6：首次真实前三候选配对及原失败保全；111原门+3完整对象门离线重核。发现并修正诊断工具的无序JSON键序问题，当前工具只将prefix键排序；测试的原029源另存冻结副本。当前具名合同59项通过，含不同PYTHONHASHSEED的三个冷进程；另保留一次遗漏声明PYTHONPATH的本地测试环境失败。没有重新运行科学计算，也没有根据结果改数值门。
- 2026-10-11，v5：首identity真实配对24门通过，但无缓存命中、未提速；原源码、协议和负结果保留，未提升默认。
- 成熟 [查询向量复用](TOPOLOGY_BLOCKER_QUERY_REUSE.md) 与 [prepared基底](TOPOLOGY_PREPARED_BASE_CACHE.md)继续使用；没有删除仍承担参考验证的实现。

## 7. 参考文献和代码

- Fischl B. FreeSurfer. NeuroImage 2012;62:774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer固定源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
- FNIT源码：`topology_ga_population.py`、`topology_candidate_cycle.py`、`topology_edge_predicate_cache.py`；诊断入口为 `benchmark_recon_topology_edge_predicate_population_prefix.py`。
- [专项冻结协议与typed收据](../../validation/recon_all/optimizations/20261011_topology_edge_predicate_population_prefix/README.md)。
