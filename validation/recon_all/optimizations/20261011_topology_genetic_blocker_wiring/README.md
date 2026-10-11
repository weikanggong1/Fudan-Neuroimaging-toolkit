# 显式库扫描后端六评分真实回归

## 1. 功能与范围

本组直接调用新`search_and_finalize_defect`，参数传至prepare/evaluator、最终评分
及额外最终几何。没有全局函数替换。完整repair/files传参和失败发布由7新增合同
核验；本组没有运行完整九缺陷或文件API的全部GA。默认仍python，numba显式可选。

## 2. 输入与输出

公开ds000114 sub07自产LH原nofix、qsphere、uint8 brain作为唯一三输入。raw base、
global priors、完整4723边、局部MRI均由本次自身MRI准备，不以冻结after初始化。
固定population5/niters0/max_unchanged10/seed1234；5次初始评分+最终1次评分，再完成
最终额外几何。两边均开启保留面索引缓存，不混state_context。
四完整NPZ保存final_candidate、final_state、search、六scores，另存完整有序控制。

## 3. 具名调用

见[七节说明](../../../../docs/recon_all/TOPOLOGY_ORDERED_BLOCKER_SCAN.md)。
新`benchmark_recon_topology_genetic_blocker_wiring.py`以两个fresh子进程串行直接
调用库API，Python→Numba；CPU84–87/总4线程、CUDA隐藏，原控制/空间/精度不变。

## 4. 原步骤与当前边界

对应原`mris_fix_topology -ga`缺陷评分/最终写回；本组只是新的显式接口绑定门，
不将5+1视为默认完整495+1。不调用原阶段程序，不读取原版winner进入算法。
原程序数值/完整整例质量和整体等效均not_assessed。

## 5. 完整真实结果

| 实际边界 | Python扫描 | Numba扫描 |
|---|---:|---:|
| fresh CLI | 118.40226643625647s | 119.88294858578593s |
| raw base自产 | 53.5198834631592s | 50.706157757900655s |
| global/shared准备 | 5.262574086897075s | 5.1318252217024565s |
| 完整有序边准备 | 6.95901576243341s | 6.661052331328392s |
| 局部MRI准备 | 2.9368889182806015s | 2.8751101437956095s |
| search+final API | 39.564361973665655s | 39.261307906359434s |
| 写产物 | 3.39090646058321s | 3.305289627984166s |

父pair总墙钟248.87594695854932s，包括两worker、准备/JIT、I/O、源码校验和
事后全部产物比较。共享负载的单次串行配对，完整CLI本组未显示提速；不推算整例。
81保存数值字段shape/dtype/bytes、四完整NPZ SHA、全部控制JSON一致。全6个double
评分与最终额外几何/持久属性均在这些保存门内。20,877次真实RNG调用的全部
(low,high,value)有序字节SHA相同：39b81ccb1b67c59ca52e377b19ba690a0fba4b59472e65c884232b7fc41d1851。
分步完整数字与RSS/源码/输入/产物绑定见metrics.json。显存allocated/reserved为null。
同源自产脑图见[完整首缺陷](../20261011_topology_genetic_blocker_scan/first_GA.png)，
该图来自默认495评分末态，不能把它当本组六评分末态或官方误差。

## 6. 版本与故障收据

实际v5科学manifestc08fea7d…；helper33a919df、dynamic74c381ac、searcha67e0979、
repair62c1fe30、files8c0d3ad0五模块实际加载SHA均核验。相对首852py冻结，仅补齐
三个必要既有FNIT依赖：next_inputs1bddff32、finish46ba4f8e、compact6add953e，
来自已有完整九缺陷冻结的逐文件SHA；没有替换已存在的其他数值模块。
`setup_failures.json`保留v1部署占位未替换、v2缺finish依赖、v3部署语法及v4预检
脚本排版四个算法前失败。v4/v5科学module/tool字节一致，只修独立preflight语法；
实际v5四入口import/签名门通过才评分，不把这些错误写成算法精度失败。
`focused_contracts.json`记录本地7新增+13既有file合同20项通过28.451s，mock仅作
调度/真实Nib文件格式及失败发布验证，不替代本组六评分真实数据benchmark。
`EXPORT_RECEIPT.json`绑定实际pair private报告/COMPLETE/EXIT与源码payload，保留
所有选中数值/布尔/null的原值及类型。原MRI/NPZ/日志/私有路径不上公共仓库。

## 7. 参考

[FNIT](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit)、
[固定拓扑源码](https://github.com/freesurfer/freesurfer/blob/d932c45b7941662ea380a05efef580568b98d41a/utils/mrisurf_defect.cpp)。
无新增依赖，使用主页现有NumPy/Numba/nibabel；无fastmath/半精度或生产默认改变。
