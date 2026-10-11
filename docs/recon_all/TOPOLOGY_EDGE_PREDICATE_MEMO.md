# 不可变球面边谓词的显式缓存实验

## 1. 功能简介

`EdgePredicateMemo` 保存同一缺陷中有向四顶点 `(a,b,v,w)` 的纯几何布尔判定。`MemoizedOrderedBlockerScan` 复用成熟 `OrderedBlockerScan` 的动态行缓冲和 `_edge_intersects_query_half`，每次仍读取当前邻接与 `valid`，按源顺序返回第一个阻挡边。查询向量、叉积、dot、`1e-7` 近零分支及完整退化回退保持原实现，不设置 fastmath 或修改线程。

跨候选共享仅限一份完整 bytes-backed 不可变 canonical。原始输入改变不会污染准备后的坐标；可变数组、普通只读 owner、部分视图、另一坐标对象和元信息变化均拒绝。字典容量满后继续计算所有 miss，只停止新增缓存项。当前是独立可选实验，没有接入生产调度，也没有更改默认后端。

## 2. Python 调用、输入与输出

```python
import numpy as np
from fnit.recon_all.topology_retessellation_dynamic import prepare_retessellation_base
from fnit.recon_all.topology_edge_predicate_cache import (
    EdgePredicateMemo,
    MemoizedOrderedBlockerScan,
)

prepared_base = prepare_retessellation_base(
    canonical=canonical_coordinates,  # float32(N,3)，完整自产 canonical RAS 坐标
    base_faces=ordered_base_faces,  # int32(F,3)，当前保留面的源顺序
    neighbors=ordered_neighbor_rows,  # 完整N行源邻接，不按面重建或排序
)
with EdgePredicateMemo(
    canonical=prepared_base.canonical,  # 准备器复制的完整 bytes-backed 只读坐标
    maximum_entries=250_000,  # 非负整数上限；满后继续原谓词求值
) as predicate_memo:
    ordered_scan = MemoizedOrderedBlockerScan(
        scan_vertices=inside_and_hull_vertices,  # 同序 int32 查询行，保留重复项
        neighbor_rows=current_neighbor_rows,  # 当前完整N行动态邻接，构造时复制
        memo=predicate_memo,  # 本缺陷独占上下文，其他缺陷必须新建
        audit=True,  # 记录每次逻辑判定，包括缓存命中，及其有序SHA
    )
    ordered_scan.sync_vertex(
        vertex=changed_vertex_index,  # 域内整数，实际邻接改变后同步
        row=current_neighbor_rows[changed_vertex_index],  # 当前完整有序行
    )
    first_blocker = ordered_scan.first_blocker(
        xyz=prepared_base.canonical,  # 必须是此memo绑定的完整数组本身
        a=query_edge_start,  # 域内整数，保留源方向
        b=query_edge_end,  # 域内整数，不排序或归一化边
        scan_vertices=inside_and_hull_vertices,  # 必须属于构造时准备的查询域
        valid=fixed_mapping_validity,  # bool(N,)，每次重新读取，不缓存
    )
    cache_report = predicate_memo.report()  # typed计数、上限、输入SHA；不代表速度
```

| 参数或输出 | 数据结构、默认值与含义 |
|---|---|
| `canonical` | 必填，C连续、完整不可变 bytes 缓冲的有限非零 native float32 `(N,3)`；原 canonical RAS 单位，不变换坐标或 signed zero |
| `maximum_entries` | 非bool非负整数，默认250000，须在int64范围；不配置估计误差或跳过规则 |
| `scan_vertices` | 必填，同序整数一维行；首次构造复制查询映射，调用时要求int32，重复行照常执行 |
| `neighbor_rows` | 必填，完整N行整数邻接；复制当前查询行，动态增删必须调用 `sync_vertex` |
| `memo` | 必填，活动的 `EdgePredicateMemo`，顶点域必须与邻接一致；不允许同时写入或fork后复用 |
| `audit` | bool，默认False；True包含序列缓冲与SHA开销，不作为普通吞吐计时 |
| `xyz/a/b/valid` | 必填，绑定坐标、两个域内整数和完整bool有效域；有效性仍作用于邻居 `w`，未改变源规则 |
| `first_blocker` | 源顺序首个 `(v,w)` 元组或 `None`；不返回候选接受或全网格质量结论 |
| `report()` | 包含 `entries`、`cache_hits`、`actual_predicate_evaluations`、逻辑调用数、命中率、插入独特键数、容量外求值次数、准备秒数和坐标SHA；达到上限后全域独特键数为 `None` |
| `close()` / `with`退出 | 清空本缺陷字典；关闭后不可使用。输入或元信息错误抛ValueError；关闭、并发或跨进程误用抛RuntimeError |

缓存不保存 marks、rip、动态邻接、first-hit、边used或种群反馈。构造、报告、扫描和同步同一上下文时禁止并发；无全局缓存、随机调用、CUDA分配或半精度。新增依赖仅复用主页Conda已有NumPy/Numba。

## 3. 命令行诊断

没有独立生产CLI。真实候选工具要求已批准的私有JSON和冻结源码，不下载或生成影像：

```bash
# CPU 0、1仅作本机示例，必须可用且与JSON cpu_affinity=[0,1]一致。
# 单臂冷进程内执行同一初态的cold/warm/audit；所有线程变量须与threads=2一致。
taskset --cpu-list 0,1 env CUDA_VISIBLE_DEVICES='' \
  NUMBA_CACHE_DIR=/data/new_memo_numba_cache NUMBA_NUM_THREADS=2 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2 \
  python tools/benchmark_recon_topology_edge_predicate_cache.py \
  --config /data/private_memo_config.json --arm memo

# 两个新进程按baseline→memo顺序完成后，才比较双方完整输出。
taskset --cpu-list 0,1 python tools/benchmark_recon_topology_edge_predicate_cache.py \
  --pair /data/private_baseline_config.json /data/private_memo_config.json /data/new_pair
```

JSON字段：`inputs`含nofix、qsphere、brain三路径；`input_sha256`逐项SHA；`source_root`为冻结根；`source_manifest`及 `source_manifest_sha256`约束全部实际FNIT模块；`protocol`及 `protocol_sha256`绑定预设输入、资源、容量和完整字段；`output`必须不存在；`hemisphere`为lh/rh；`threads`为2或4；`cpu_affinity`为实际CPU编号列表；`maximum_entries`为上述上限；`numba_cache_dir`为本臂独立空缓存。两个进程使用相同线程和亲和性，GPU隐藏，不重用另一臂JIT缓存。

工具以临时factory接入实验scanner，两个臂同样观察外层完整scan时间，`global_function_replacement=True`，总在finally恢复原类。`--pair`为每个新子进程设置相同的Numba、Torch/OpenMP、BLAS和ITK声明线程预算；单臂调用也要求各线程环境一致，并记录Torch/Numba实际掩码及恢复状态，未声称观测所有OS线程。输入准备只用自产三文件；另臂输出在双方完成前不读取。返回/写出cold、warm、audit完整候选NPZ、持久状态NPZ、控制JSON、哈希收据、计时和RSS。完整CLI计时包含导入、校验、原始输入准备、三次候选、输出写入和退出；事后SHA、NPZ及控制比较读出另计，不是原始T1整例。

比较器在读取两份完整收据后，强制检查36个候选字段、40个持久状态字段、6个控制字段与恰9项artifact。两臂共同遗漏字段也会失败。实际诊断入口、所有FNIT模块、全部已导入benchmark helper的相对路径、origin与SHA必须属于同一冻结manifest；候选状态和payload两个helper在计算前还核函数定义位置，结束时复核。缺字段、错绑来源或不完整收据明确拒绝。

## 4. 对应原软件

这是 `mris_fix_topology -ga` 内部有序重连的球面边相交判断，没有独立官方CLI。原有公式仍由项目成熟谓词实现；只缓存同一固定输入的布尔值，没有复制或运行原版C++。对应源码为固定FreeSurfer版本的 [mris_fix_topology](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mris_fix_topology) 及 [mrisurf_defect.cpp](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)。完整原软件对照只能在独立benchmark路径运行。

## 5. 最新精度与耗时

本轮已完成本地合同：不可变域、方向键、近零/退化、极端float32尺度、容量外完整求值、动态删加与valid反转、成熟完整两轮重连及审计序列。小固定几何用于验证缓存语义，不能作为真实MRI benchmark或提速证据。

已完成冻结v5的真实首identity候选配对。输入是既有公开T1的三项FNIT自产上游：`orig.nofix`、`qsphere.nofix`和`brain.mgz`，没有读取旧候选或参考结果作为输入。科学源码固定于Git `e9e89cac07cc65ef37f8c3223e53f03968f2af05`，只覆加本模块。两臂在同一共享主机上顺序启动，各自使用空Numba缓存、相同两线程CPU预算，并隐藏CUDA；每臂从相同初态独立执行cold、warm、audit，缓存不跨这三次重复共享。

完整24项比较通过：36项候选数组、40项状态数组的完整字段集、shape、dtype和bytes，6项控制字段，三轮跨臂与自身重复，以及有序谓词bool序列均一致。随机回调调用为0，输入前后SHA不变；两臂实际载入的65个FNIT模块和10个benchmark模块均来自同一冻结包，线程策略已恢复。另一次独立导出重核全部门后才发布[机器结果](../../validation/recon_all/optimizations/20261011_topology_edge_predicate_memo/metrics.public.json)和[来源与产物收据](../../validation/recon_all/optimizations/20261011_topology_edge_predicate_memo/execution_receipt.public.json)。

| 本次实测范围 | 未缓存基线 | memo | 观察 |
|---|---:|---:|---|
| 完整冷CLI，含导入、校验、自产准备、三次候选及读写/退出 | 130.129983 s | 131.456263 s | 慢1.02% |
| cold完整候选API，含准备、排序、闭环 | 26.332810 s | 27.537387 s | 冷启动单独列出 |
| warm完整候选API | 8.434330 s | 9.535094 s | 慢13.05% |
| warm候选闭环 | 1.642178 s | 2.863828 s | 未见收益 |
| warm完整扫描API | 0.242085 s | 0.540042 s | 未见收益 |
| warm重连子时钟 | 0.419643 s | 1.579466 s | 与上层时钟嵌套，不相加 |
| warm候选API及NPZ/控制写出 | 11.614411 s | 12.889571 s | 包含实际写出 |
| 进程峰CPU RSS | 1,796,235,264 B | 1,836,576,768 B | 增加40,341,504 B |

三轮各有351,038次逻辑谓词调用，memo每轮命中0次、实际求值351,038次、存入250,000项、cap外求值101,038次。cap之后未保存全部键，因此完整独特键数仍是`null`，不能把调用数当作独特键数。这个候选没有省掉几何计算，字典查找和写入增加了开销。共享环境中的单次顺序配对不能建立稳定吞吐结论，本次也没有支持接入生产的提速证据。

本次基线是同源码的成熟FNIT未缓存路径，不是新跑的官方结果。完整495次搜索、真实跨候选缓存复用、最终GA写回、网格质量和原始T1整例仍为`not_assessed`；未生成新的整例脑图。GPU未使用，allocated/reserved为`null`，不是0。生产路径和默认后端保持原样。成熟未缓存版本的既有真实证据见[查询向量复用](TOPOLOGY_BLOCKER_QUERY_REUSE.md)，不能把其耗时当成本实验结果。

## 6. 更新与benchmark记录

- 2026-10-11，v5真实记录：首identity配对、完整24门和独立typed导出通过；命中0、冷CLI和warm完整候选API均未提速。保留完整负结果，未提升为生产默认。冻结协议的准备状态保留原字节，实际完成状态见执行收据。
- 2026-10-11，本地合同：50项成熟/新模块合同和3项导出合同通过；这些小固定几何及伪造schema收据不属于MRI benchmark。
- 既有有序扫描、查询向量复用和不可变基底实现保留，分别见 [扫描](TOPOLOGY_ORDERED_BLOCKER_SCAN.md) 与 [基底](TOPOLOGY_PREPARED_BASE_CACHE.md)。未替代的实现继续承担当前生产及参考验证，不删除历史数据或报告。

## 7. 参考文献与源码

- [本模块](../../src/fnit/recon_all/topology_edge_predicate_cache.py)、[诊断工具](../../tools/benchmark_recon_topology_edge_predicate_cache.py)、[合同测试](../../tests/recon_all/test_topology_edge_predicate_cache.py)。
- [FreeSurfer固定源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
- Fischl B, Liu A, Dale AM. Automated manifold surgery: constructing geometrically accurate and topologically correct models of the human cerebral cortex. IEEE Transactions on Medical Imaging. 2001;20:70–80. [DOI](https://doi.org/10.1109/42.906426)。
