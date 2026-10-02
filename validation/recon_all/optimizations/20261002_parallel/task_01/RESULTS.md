# 任务01交付：半球调度与真实表面链比较

实现默认保留串行 `hemisphere_workers=1`，新增可选2个独立exec worker；四组阶段有发布屏障，私有目录隔离，shared MRI按原串行语义合并。最终顶点指标按左右串行计算。Python、CLI、batch及输出结构见 [README.md](README.md)，共享写入审计见 [AUDIT.md](AUDIT.md)。没有新增依赖、权重或再分发原软件代码。

## 真实配对结果

在gpucw1固定GPU UUID上，从冻结自产MRI前缀重新生成surface、registration、三套annotation、最终white/pial与指标；启动前清空surf/label/stats。总线程预算4；基线串行4，双侧各2。输出准备复制在计时前完成，组内复制、exec、同步、发布、清理计入。

| 阶段 | 串行秒 | 双半球秒 | 串行/双半球 |
| --- | ---: | ---: | ---: |
| surface及球面 | 1348.947 | 689.133 | 1.957 |
| sphere registration | 247.315 | 253.218 | 0.977 |
| 三套annotation | 190.335 | 102.394 | 1.859 |
| 最终white/pial放置 | 734.396 | 365.866 | 2.007 |
| 完整受影响表面链，含缺陷体积、串行指标及调度开销 | 2530.086 | 1420.959 | 1.781 |

单次AB顺序的共享服务器观测，未完成此长链BA重复。该结果不代表原始T1整例端到端速度。之前独立metrics AB/BA显示并行较慢，已据此选定指标串行；独立annotation同输入AB为1.692倍且六annot逐字节一致。

原始计时与所有源/输入/程序哈希：[chain_AB.json](chain_AB.json)；组复制、发布、清理、worker总时长/跨度/重叠：[chain_timing.csv](chain_timing.csv)；摘要：[chain_summary.json](chain_summary.json)。

## 精度与质量

- 双侧最终坐标及有序面相同；orig/white/pial/sphere.reg闭合、Euler=2、坐标有限，white/pial单张网格自相交0。
- 六annot文件及全部检查的cortex标签一致，最终厚度严格相同；最终面积、曲率、TH3顶点体积通过原有算子容差。surface.defects标签/affine完全一致，压缩文件字节不同。
- 逐区比较覆盖278条记录：顶点数、white面积、厚度均值/标准差、no-TH3灰质体积差异全为0。统计来自本次重算表面与注释，不是复制历史stats。
- 双侧sphere及sphere.reg的负向面与零朝向面均0，球面半径约100 mm。
- 全量cortex white/pial两网格面配对按现有1e-6平面容差计入接触，左117、右228对，两模式完整面配对列表相同；对应面逆向棱柱左417、右441也相同。这些诊断与单张表面自相交不同，未宣称两网格交叉为0，也未判定原算法质量优于原软件。
- 91项实际自产受影响文件检查：79通过；4张white.preaparc.H/K的同算子容差映射缺少测前预声明证据，仅列后验诊断（诊断在white.H/K容差内，状态not_assessed，最大差9.536743e-5），不计验收通过；8项smoothwm H/K/K1/K2中间曲率图因没有预声明非零门槛，按0容差失败，最大差0.000333786；无缺失、无对应关系阻断。保留失败，不追加事后容差。总体数值回归标记failed_or_not_assessed，严格逐字节复现failed，科学等效not_assessed。
- 本次阶段目录在138项全流程清单中仅97项存在；共有MRI来自冻结前缀，exvivo/ribbon/体积投影/最终stats等没有重算，不能当作138项整例验收。

逐文件冻结门槛与单列后验诊断结果：[chain_affected_numeric.json](chain_affected_numeric.json)；完整质量、所有交叉面对及逐区结果：[chain_quality_and_regions.json](chain_quality_and_regions.json)、[chain_regional_differences.csv](chain_regional_differences.csv)。中间sphere线搜索的overflow/invalid警告在串行和并行均保留于日志，最终finite与fold检查上述通过；未更改成熟数学实现掩盖警告。

## 资源、线程与采样

11个权重、102个资产、14个程序及2个原始输入大小/SHA-256全部匹配协调者冻结清单，输入检查点未变化。授权资源保持原位置，不包含影像、权重和许可证内容。

GPU进程树观测峰值串行1,992,294,400字节、并行5,922,357,248字节；父与所有活跃子进程同一采样累加，没有相加独立历史峰值。串行3254次采样、2次失败、最大间隔14.18秒；并行2155次采样、0失败、最大间隔2.865秒，目标0.5秒。不能将观测峰值当连续上界。GPU外部进程显存最大值串行0.843 GB、并行13.602 GB，两模式负载不同，全部同期样本在原报告中；未停止外部任务。

CPU旁路记录仅覆盖启动后的一段3408.2秒，6066次样本，最大间隔2.762秒；1分钟loadavg范围22.03–51.56、平均35.47，affinity包含topology固定1核及其他128核可见范围。OS线程数包含闲置运行库线程，不能当计算线程数。[chain_cpu_load_summary.json](chain_cpu_load_summary.json)、[压缩原始观测](chain_cpu_load.json.gz)。

## 复现与版本绑定

实测冻结源码commit `db7af9b1e1ec67a12e6c6c253b212589c8d0c663`，当前候选新增失败收尾和API边界修复，数值函数AST独立核验不变：[numerical_source_continuity.json](numerical_source_continuity.json)。不得把冻结测量标成最终候选已完成整例运行。报告来源修订：别名脚本首次记录于测后a2ef51b，冻结AUDIT与chain_AB的numeric_tolerances均无该映射。原83项通过的分类不成立，修订为79通过/4后验not_assessed/8失败；存量数值未改，未重跑分析或性能。原报告与执行脚本SHA-256、修订脚本SHA-256及测量commit分别保存在chain_affected_numeric.json的report_correction；原性能/输入/source SHA-256保留。专项CPU测试50项及14子测试通过：[cpu_tests_boundaries.log](cpu_tests_boundaries.log)。

```bash
# 在对应冻结源码目录运行；所有资源必须先按固定清单验证。
# GPU_UUID使用清单指定的完整UUID；Python解释器使用项目Conda环境。
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export PYTHONPATH="$SOURCE_DIRECTORY/src:$SOURCE_DIRECTORY/validation/recon_all/optimizations/20261002_parallel/task_01"
flock -x /tmp/fnit-recon-five-20261002-gongwk.gpu.lock \
  "$CONDA_PYTHON" validation/recon_all/optimizations/20261002_parallel/task_01/benchmark_hemi.py \
  --checkpoint "$FROZEN_MRI_SUBJECT" --output "$EMPTY_PAIR_OUTPUT" \
  --assets "$VERIFIED_ASSETS" --binaries "$VERIFIED_NATIVE_BIN" \
  --device cuda:0 --operation chain --order AB \
  --commit db7af9b1e1ec67a12e6c6c253b212589c8d0c663 \
  --resources-manifest "$FROZEN_RESOURCES_MANIFEST"
# 计时完成后执行CPU质量诊断，不计性能。
export CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1
"$CONDA_PYTHON" validation/recon_all/optimizations/20261002_parallel/task_01/analyze_chain.py "$EMPTY_PAIR_OUTPUT"
"$CONDA_PYTHON" validation/recon_all/optimizations/20261002_parallel/task_01/assess_chain.py "$EMPTY_PAIR_OUTPUT"
```

协调者负责合并最新main及其他四项优化后，在两例原始T1空目录分别完成CLI/Python API整流程、官方参考精度/时间及最终主分支验证。本任务未推送main，未动其他任务目录；完整整例、官方等效与最新候选GPU边界整合验证仍待协调者完成。
