# SynthSeg 单层复用 CPU 接入候选（2026-10-06）

## 1. 功能与状态

独立候选将既有同层逐位一致的FNIT自有copy/SGEMM胶水接入33类网络最后一级。**本地39项mock守卫/缓存合同已通过；新生产实际编译、短数值合同和whole尚未运行，不能接入main或称端到端通过。** 本次没有新MRI/native/GPU/列copy/SGEMM调用。

新三文件是 `cpu_columns.py`、`_cpu_columns_build.py` 和4385B `_columns_reuse.cpp`；现有 `cpu_conv.py` 只在CPU/F32/eval/no-grad/oneDNNFalse的已标记层进行lazy资格检查，None继续原slab。`segment.py` 仅在33类网络up[3].conv0加metadata标记；CUDA/训练路径不导入helper。C++与已验收私有源逐字节相同，数值forward尾AST相同；不改原通用slab和低内存防崩路径。

## 2. 输入、输出、Python与参数

用户调用及完整输入/输出/参数见 [七节功能说明](../../../docs/synthseg/CPU_COLUMNS.md)。公共函数、默认TF32/False/None、CSV/标签/几何不改，无新算法选项。普通SynthSeg CPU关闭oneDNN，因此本真实shape原图+flip两pass可激活；公开Plus/parc/fast CPU默认oneDNNTrue不激活；皮层ParcUNet不标记。调用方显式关闭oneDNN时，只有同已验权重和shape的共同33类末层可资格检查，未知条件均保持原路径。

资格须是严格CPUInferenceConv3d、普通Tensor/Parameter、无forward-AD/懒negative/conjugate视图、CPU FP32输入 `[1,72,192,224,256]`、连续参数 `[24,72,3,3,3]` / `[24]` 且实际值SHA相同、8线程、无autocast/梯度/hooks、同stride/dilation/padding/groups。每个合资格CPU层重新核参数约0.187MB，捕获不递增_version的.data改值；GPU构造只加marker，不读HDF5做额外SHA。

`CXX`、`FNIT_SYNTHSEG_CPU_CACHE`及XDG私有缓存规则在功能页列全。缓存只存小.so/manifest和锁，columns最大6,242,697,216B只在单次层调用内存在，不被engine/model缓存。provider只能来自已经加载的Torch handle/global同地址MKL LP64及SHA，禁止libblas/mkl_rt/provider替换或隐藏符号offset。

## 3. 本地合同与待执行命令

实际本地命令使用目标Torch2.5.1环境并在harness中显式绑定候选src，覆盖测试收集路径、源SHA和flags；没有Compiler/MRI性能意义。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
  python validation/smri_cpu/seg_columns_integration_20261006/run_local_guards.py
```

实际39 passed in3.05s见 [LOCAL_GUARDS.log](LOCAL_GUARDS.log)，source/flag前后门和过程时钟见 [原receipt](LOCAL_GUARDS.json)。positive参数SHA资格在local合同中mock；未知真实参数rawhash反例、实际Tensor负/共轭view及dual/subclass守卫另测。cache编译/dlopen是stub，不得把它标为真实.so/安装验证。38项旧receipt/log保留为v1；此前一次因旧editable安装路径在pytest collection退出2，零测试/科学调用，详见 [历史](HARNESS_HISTORY.json)。

`prepare_plan.py`仅stdlib读取冻结源/已有标量报告和AST，生成 [PLAN](PLAN.json) / [STATIC_CHECKS](STATIC_CHECKS.json)，不import Torch、不派发。`whole_worker.py`、`whole_queue.py`是待审代码：显式approved参数只是程序门，不能代替root授权。短合同四份实际worker已独立冻结，见 `phase1_bindings.py`、`load_cache_interface.py`、`check_cached_contracts.py`、`run_phase1.py`；当前尚未获实际编译/数学执行授权。metadata先在全新私有cache编译/ABI/provider-load（copy/SGEMM为0），短合同fresh进程复用该cache且禁止再编译。原六数值/13copy/23fallback合同的test body逐字节相同；只有loader/setup替换，私有兼容shim将生产None映射到旧fallback sentinel，生产不变。当前不能执行whole。

另有4项纯队列控制mock合同通过，见 [QUEUE_CONTRACTS](QUEUE_CONTRACTS.json)。成功B1/B2/A2必须立即与A1的同名完整gzip/CSV SHA相同，门过后才能派下一臂；SHA、cold/warm compiler计数或worker后置门失败均先持久失败QUEUE及已完成worker的原exit0，再停止余臂。A1仅生成参考，comparison_executed=False。mock不派真实child，不import Torch，不做MRI/编译/数值。

## 4. 对应原步骤与构建

原软件这一个内部卷积没有独立CLI。矩阵和bias规则见 [已通过的单层ABBA](../seg_columns_real_layer_v2_20261006/README.md)。homepage Conda已有Torch2.5.1/GCC11，无新增依赖；package-data和MANIFEST声明自有cpp，wheel不带动态库、模型或数据。新build cache实际编译和安装还未评估。

构建先核Linuxx86_64、Torch2.5.1/ABI0、TorchCPU314MB和8头文件SHA、公共LP64 provider及地址，再核GCC11编译器，创建本人0700cache/0600lock与artifact并atomic replace。编译120秒、锁15秒，失败同key不重复编译；未知准备条件在数值前None回退。数值engine调用后异常传播、没有旧卷积重试。源码/header/provider及compiler probe每个合资格调用仍做，cold/warm whole必须包含这些成本。

## 5. 已有结果与新whole预声明门

[SOURCE_CONTEXT](SOURCE_CONTEXT.json)来自本轮实际canonical main/INDEX及14源/原T1/资产只读核验；不能拿历史snapshot当baseline。正式旧native结果来自 [同node7/8物理核记录](../seg_memory_20261005/OFFICIAL_NODE7_FAST_BA_REPEAT.public.json)，PLAN绑定原seg33重复结果map/CSV SHA，**只在新输出保存后独立评分，不进生产输入，不重跑native**。

旧私有单层ABBA三个实际完整264,241,152值bits0；约15.39→5.72秒、RSS11.40GB，不能标为新生产whole性能。旧完整33约112.95秒/native55.05秒仍未速度门；native373秒异常原样保留且不用作分母。当前没有新完整脑图，既有图 [逐标签](../seg_memory_20261005/case02_cpu_labels.png) 只属于原版本。

后续两阶段分别审查/授权：

1. 新私有contract cache一次metadata编译/加载，随后六个固定真实权重短合同、copy13/poison/紧凑末次M/signed0及fallback/异常守卫；无MRI，原14-plane reference，首差停止。其worker已冻结待root授权。原test body SHA `ae4aa4bf74342bf265115b3749557a79b05a812effcce17e277ee2e963123b0c`，seed20261006/原六shape不变；失败保留receipt并停止后续，没有科学重试。每个worker释放共同锁；metadata180秒、contract240秒、AS8e9/RSS32e9、outer23000秒。
2. CPU33 A1原/B1cold/B2warm/A2原共4新进程，每arm600秒、共同CPU8锁、同8物理核、32e9 AS/RSS，原T1+空输出。B1是新独立whole cache（保留short cache，不删除），B2同cache的新进程；每候选实际两pass/两层命中，28copy/28SGEMM。clock含hash/compiler/compile/构造/save/收尾，另列API/构造/save。GPU默认True普通33原/新AB共2个新进程、各300秒、同H100UUID与20e9预算，必须实际零optional-module import/compile/copy；GPU监测不能为了查状态先import helper。

全标签、affine/完整header/extensions/dtype、数值CSV和CSV文件SHA严格相同；记录压缩图SHA。完整输出先保存再断言diagnostic计数。GPU全allocated/reserved exact≤20e9，另记录本人process-tree driver采样，最大gap/failure/zero明确，采样不是绝对峰值；共享GPU时间仅观察。全部源/资源/参数/精度/observer正常异常收尾门明确，首失败停余臂，不科学重试或放宽门。

默认CPUparc/fast不触发候选，不重跑上一版无影响whole；其oneDNN策略不为了提速更改。已有False/None和robust边界保留，当前普通--parc不是robust SynthSeg+。common锁每arm释放，outer总deadline23000秒不重置。

## 6. 更新与后续

- 已接受同层阶段提交7b69bc93；对应四原receipts与三位门保留，本candidate不重新跑层ABBA。
- 接入local第1次收集错误0科学、第2次37合同、第3次38合同（新增GCC版本反例）、第4次39合同（实际懒view守卫）；旧38source-bound receipt/log保留，源码变动只在新freeze。
- 新准备冻结保留70ad6537的原PLAN/manifest Git来源；生产17源未改变。common四source由当前canonical/INDEX只读逐SHA绑定，`Conda_CXX`路径来自旧真实interface argv，实际新编译器identity将在metadata cache-key中核对。whole每个成功arm立即做原gzip/CSV SHA严格门，失败QUEUE先持久再停余臂。
- 下一候选可静态研究其他groups1/k3的同原slab workspace复用，但本freeze只72→24一层，不扩shape、layer或新BLAS/低精度。
- 本地guard通过与prepared PLAN不能替代实际编译/完整CPU/GPU/Conda安装；当前这些状态保持not_assessed。

## 7. 来源及许可

新C++、缓存与守卫是FNIT自有代码；仅复用用户已安装Torch头/库和公共SGEMM，不再分发Torch/MKL/libgomp或原软件源码。见 [own notice](../../../src/fnit/synthseg_parc/CPU_COLUMNS_NOTICE.md)、[PyTorch2.5.1 Slow3d](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/ConvolutionMM3d.cpp)、[CPUBlas](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/CPUBlas.cpp)、[SynthSeg](https://github.com/BBillot/SynthSeg)。权重继续外置原条款；原T1为CC0 OpenNeuro来源，私有影像/数组/动态库不进入本leaf。
