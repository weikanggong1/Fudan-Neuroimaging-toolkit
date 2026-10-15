# 完整自有 GA 的 recon-all 接线

## 1. 功能简介

`backend="python", topology_backend="python-ga"` 显式选择安装包内的完整有序遗传拓扑算法，替代拓扑阶段程序。Python、NumPy、Numba 的顺序更新保留；已有 GPU 体积、球面和指标函数继续使用。数字拓扑的旧选择 `topology_backend="python"` 和现有默认值保持。

```mermaid
flowchart LR
  V[自产 MRI / WM / filled] --> N[orig.nofix / qsphere.nofix]
  N --> G[完整有序 GA / SOAP / 压缩]
  G --> P[ORIGINAL premesh / 原域缺陷标签]
  P --> R[标准三轮 remesh / 完整相交清理]
  R --> Q[最终 orig 网格与几何检查 / SHA]
  Q --> W[发布 / white.preaparc / sphere.reg / white / pial]
```

已有冻结单半球运行完成九缺陷与最终 `orig`，与旧自有基线的404项数组相同。后续 cbaffa 合并配方的[两例原始 T1 整例及 v2 事后评估](../../validation/recon_all/optimizations/20261016_gpucw1_cbaffa_two_case/README.md)均退出 0、输出 138/138，生产网格门通过；两例分区、表面距离和脑区统计评估已完成，严格逐文件复现分别为 14/138、23/138，整体指标等效仍未判定。冻结阶段的同输入结论与这次整例分别记录。

## 2. Python 调用、输入与输出

```python
from fnit.recon_all.native_free import run_recon_all_python

reconstruction_report = run_recon_all_python(
    t1="/data/sub01_T1w.nii.gz",  # 原始三维单幅 T1w NIfTI
    subject_dir="/data/subjects/sub01",  # 不存在或为空的本次输出目录
    weights_dir="/data/fnit-weights",  # 已按项目清单校验的网络权重
    assets_dir="/data/fnit-assets",  # 已校验的图谱、模板和标签表
    device="cuda:0",  # 显式目标 GPU；完整 WM/MNI 接口需要 CUDA
    threads=4,  # 整例 CPU 总预算；双侧各两线程，GA 不另开四线程
    backend="python",  # 完整自有配方，不发现或运行原版阶段程序
    topology_backend="python-ga",  # 显式完整有序 GA，不被配方解析覆盖
    n4_backend="simpleitk",  # 完整通用库 N4，保留已知强度差异记录
    n4_execution="in-process",  # 本次 N4 在调用进程执行
    input_execution="serial",  # 输入阶段沿用串行策略
    profile_stages=True,  # 测评同步明确 GPU；生产可设 False
    cuda_allocator_cache="auto",  # 保留既有低显存策略和已初始化 API 分配器
)
```

| 参数 | 默认值、格式与限制 |
| --- | --- |
| `t1` | 必填影像路径；原始三维单 T1，不接受官方中间结果替代 |
| `subject_dir` | 必填目录；不存在或为空，禁止覆盖既有被试结果 |
| `weights_dir`、`assets_dir` | 必填声明资源目录；预检大小/SHA 和必要文件，不重复下载 |
| `device` | 默认 `cuda:0`；完整 Python 配方要求带编号的 CUDA 设备，CPU 有序核仍允许 |
| `threads` | 默认 4、整数且至少 2；两个隔离半球 worker 各 `threads//2`，奇数余量不分配给 GA |
| `backend` | 公共 API 默认 `native`；本调用显式 `python`，选择完整自有配方 |
| `topology_backend` | 公共 API 默认 `native`，在 Python 配方中解析为数字 `python`；显式 `python-ga` 使用完整自有 GA，无原版拓扑程序 |
| `n4_backend`、`n4_execution` | Python 配方默认 `simpleitk`/`in-process`；完整 Torch/isolated 的既有显式选择仍保留，不能据 GA 接线改变 N4 算法 |
| `input_execution` | 默认 `serial`；既有 `parallel` 仅允许 SimpleITK、四线程及其完整输入调度规则 |
| `profile_stages` | 默认 False；True 同步显式 GPU 并记录等待，整例墙钟仍包含加载、传输及读写 |
| `cuda_allocator_cache` | 默认 `auto`；既有 initialized API 策略保持，不以 allocated/reserved 为零宣称零显存 |

其余参数与[完整 Python 配方](PYTHON_SURFACE_PIPELINE.md)、[主接口](README.md)一致。本次只新增已有 `topology_backend` 的取值，不增加一套重复算法。底层文件接口的全部 GA 参数、默认值及返回字段见 [GA 文件接口](TOPOLOGY_GENETIC_FILES.md)。

阶段输入为同一自产链的 `H.orig.nofix`、`H.qsphere.nofix`、`brain.mgz` 和 `wm.mgz`。前两者必须具有相同顶点数和有序面，坐标为 surface RAS/mm；brain 是 conformed uint8 XYZ，供强度评分和 RAS→voxel 矩阵；WM 几何供表面写出头。原始 nofix 域的缺陷编号和压缩后 premesh 域分开，不能把压缩标签配到原网格。

文件接口先在新目录 `scripts/H.topology.genetic/` 完成 GA、三轮 remesh、`source_numba` 相交清理与最终质量门，输出真实 `premesh`、原域 `defect_labels` 和 `orig`。该配方显式设置 `face_detection_cache=True`，每缺陷复用不可变保留面索引，并选择 `blocker_scan_backend="numba"`，将完整有序阻挡扫描移入项目已有谓词的编译循环。动态坐标、评分、候选域和随机流仍完整计算；没有开启 fastmath 或降低精度。底层独立文件 API 默认关闭面缓存、使用 `blocker_scan_backend="python"`，便于同输入对照。

调度核对完整/质量状态、实际缓存和扫描参数、报告及三文件 SHA，随后发布标准 `surf/H.*` 文件；不重复执行 remesh。缺少实际后端记录或返回其他后端时直接失败。`H.topology.genetic.publish.json` 保存相对被试目录的路径、SHA、半球线程、质量及实际缓存/扫描选择，表面阶段返回 `topology_genetic_publication`。整例返回 JSON 字典及现有输出路径清单；各步骤耗时为嵌套时间，不与父阶段重复相加。

失败抛异常并保留私有诊断，不发布未验收 `orig`。双侧 GA 状态目录按半球保留层级；同名 `report.json` 不再扁平化覆盖，另一半球写入该目录会在发布前失败。共享 `surface.defects.mgz` 仍由父进程按 LH→RH 顺序投射。

## 3. 命令行调用

```bash
reconstruction_arguments=(
  /data/sub01_T1w.nii.gz             # 原始单 T1
  /data/subjects/sub01               # 空输出目录
  --weights-dir /data/fnit-weights   # 项目权重
  --assets-dir /data/fnit-assets     # 声明图谱与模板
  --device cuda:0                    # 明确目标 GPU
  --threads 4                        # 整例总 CPU 预算
  --backend python                   # 不执行原版阶段程序的完整配方
  --topology-backend python-ga        # 完整自有 GA，CLI/batch 保留该选择
  --n4-backend simpleitk              # 完整通用库 N4
  --profile-stages                   # 测评显式 GPU 同步
)
fnit-recon-all "${reconstruction_arguments[@]}"
```

batch 的 `run_recon_all_python_batch(..., backend="python", topology_backend="python-ga")` 使用相同配方，各设备只运行一个被试。它不把显式 GA 改回数字拓扑。

整例计时复用现有包装脚本，显式选择同一 GA 配方：

```bash
benchmark_arguments=(
  --t1 /data/sub01_T1w.nii.gz         # 原始单T1，不使用官方中间结果
  --output-root /data/new-ga-run      # 尚不存在的本次整例目录
  --weights-dir /data/fnit-weights    # 已校验网络权重
  --assets-dir /data/fnit-assets      # 已校验图谱和模板
  --backend python                   # 独立Python配方
  --topology-backend python-ga        # 包装脚本保留GA，不覆盖成数字拓扑
  --device cuda:0                     # 明确目标GPU
  --threads 4                        # CPU总预算，双侧各两线程
  --code-version '<Git commit>+manifest:<SHA256>' # 实际安装源码及包装脚本的版本
)
python tools/benchmark_recon_torch_end_to_end.py "${benchmark_arguments[@]}"
```

包装脚本输出原始被试目录、`benchmark.json`、完整CLI日志及执行启动审计；分别记录执行、完整性、严格复现、优化回归与整体等效状态。整例CLI包含导入、预检、传输和读写；harness另含源码与资源哈希。异常保留部分报告并返回非零，不补跑或把检查点解释为整例成功。

## 4. 原软件调用

以下只在隔离 benchmark 中生成参考，生产不启动这些程序：

```bash
mris_fix_topology -ga -seed 1234 -threads 1 -mgz \
  -sphere qsphere.nofix -inflated inflated.nofix \
  -orig orig.nofix -out orig.premesh subject lh
mris_remesh --remesh --iters 3 \
  subject/surf/lh.orig.premesh subject/surf/lh.orig
mris_remove_intersection subject/surf/lh.orig subject/surf/lh.orig
```

安装包文件接口、SHA 发布和半球路径审核属于 FNIT 调度，没有独立对应的 FreeSurfer 命令。GA 的随机流、搜索控制、顶点剔除及完整收尾对应固定源内部流程；程序版本、参数和实际线程以各运行 manifest 为准。

## 5. 当前精度、耗时与脑图

| 验证范围 | 已完成的实测 | 可以得出的结论 |
| --- | --- | --- |
| 同输入候选/写回 ABBA | 162 项完整状态和控制无差；CLI 中位数 76.883→72.862 秒 | 保留面索引复用的阶段回归通过，非整例收益 |
| 同输入完整默认首缺陷 | 495 搜索评分＋1 最终评分、29 轮、随机流、80 字段和三份检查点相同；CLI 2263.423→1329.629 秒 | 共享主机跨时段观察；非 ABBA、非全九缺陷或原始 T1 整例 |
| 原生参考同输入重复 | 两次完整九缺陷 premesh 数值坐标/有序面一致；程序约 172.849/155.905 秒 | 原生参考可重复性证据；不同时间/实际原生单线程，不能与首缺陷直接计算速度比 |
| 当前面缓存库入口，同输入六评分 | 81项数组、四份检查点及20,877次随机调用相同；API 48.945→36.928秒，冷CLI 124.854→114.997秒 | 实际库入口的阶段回归；非默认完整搜索、非ABBA、非整例 |
| 第四缺陷同输入完整候选，有序扫描 | 暖闭环6.779→3.825秒，72字段及2,656,862次完整谓词序列相同 | 本候选阶段提速；不能外推完整GA或整例 |
| 首缺陷完整搜索，有序扫描 | 495次搜索＋最终评分、80字段/三检查点、29轮控制及随机调用数相同；API 1252.549→1349.788秒 | 不同时段共享环境；本次完整阶段未观察到提速，未捕获逐次随机值序列 |
| 新库入口同输入六评分，有序扫描 | 81字段/四检查点、20,877次随机调用的bounds/value顺序SHA相同；搜索＋收尾39.564→39.261秒，冷CLI118.402→119.883秒 | 直接库参数贯通；非完整默认搜索，不能宣称CLI或整例加速 |
| 当前查询边向量复用，完整首缺陷 | 495次搜索＋最终评分、80字段/三检查点与29轮控制相同；冷CLI1357.057→1217.854秒 | 单次、不同共享负载时段的阶段观察；非整例提速。第四候选暖闭环改善，冷启动及短搜索负结果一并保留 |
| 既有生产文件接口/配方 | 文件接口13项合同；配方/发布30项测试及121项子测试通过，24.11秒；离线wheel独立安装/import/fresh CLI均通过 | 保留其原版本；不能代替真实 MRI、当前GPU前向或完整运行 |
| 当前有序扫描配方接线/安装 | 发布合同9测试/23子测试通过，15.90秒；本地和服务器实际安装十二模块的source/wheel/installed SHA相同，服务器安装与入口17.598秒 | 未运行本安装产物的MRI/GPU；不是整例时间 |
| GA 原始 T1 首次启动 | 公开 sub06 在 N4 因缺少 SimpleITK 失败；CLI 66.778秒，API 61.827秒，包装含资源校验81.065秒 | 保留失败收据；不是整例时间或质量通过 |
| 冻结单半球九缺陷与最终 orig | 九缺陷CLI15196.729秒，后续orig CLI183.192秒；404项数组与旧自有基线相同；最终网格Euler2、单连通、相交0 | 对应旧Python扫描/面缓存冻结版本；与原生网格顶点数不同，不能逐索引比较，非当前配方整例提速 |
| 后续合并源码离线安装 | Git `342aa4dd`，24模块source/wheel/installed SHA相同；构建2.401、安装0.766、入口10.242秒 | 完整EM和可选GPU距离接口导入通过；不初始化CUDA、不运行影像 |
| 隔离worker守卫后的离线安装 | Git `9292a0a7`，29模块source/wheel/installed SHA相同；构建2.294、安装0.849、入口17.429秒 | GCA、WM、Talairach、Torch N4的fresh CLI继承生产禁令；36项合同通过，不运行MRI |
| cbaffa 配方原始 T1 整例 | 两例退出 0、各 138/138 输出、生产网格门通过；完整外层墙钟 19,448.046／12,094.283 秒，详见[两例报告](../../validation/recon_all/optimizations/20261016_gpucw1_cbaffa_two_case/README.md) | 严格逐文件诊断 14/138、23/138；十分钟目标未达到。整体指标等效、FNIT 自身 20 GB 峰值和同环境提速未判定 |

真实脑图与全部 496 项比较见[首缺陷完整报告](../../validation/recon_all/optimizations/20261011_topology_genetic_face_cache/README.md)，阶段 ABBA 见[保留面索引报告](../../validation/recon_all/optimizations/20261011_topology_face_detection_cache/README.md)，原生重复见[参考重复报告](../../validation/recon_all/optimizations/20261011_topology_native_repeat/README.md)。首缺陷脑图显示自产几何及索引误差，不能称官方整例误差图。既有数字拓扑整例的局部形状/球面失败保留在[旧版完整报告](../../validation/recon_all/optimizations/20261010_python_profile/README.md)，不改标为本次 GA 配方结果。

GA 新接口没有新增依赖；NumPy、Numba、SciPy、nibabel 和 PyTorch 已由主页 Conda/包安装声明。[2026-10-11 面缓存配方的合同及安装报告](../../validation/recon_all/optimizations/20261011_python_genetic_profile_cached_route/README.md)绑定当时的代码及wheel哈希；[首版接线记录](../../validation/recon_all/optimizations/20261011_python_genetic_profile/README.md)保留原始版本。面缓存科学阶段见[实际库入口报告](../../validation/recon_all/optimizations/20261011_topology_genetic_face_cache_wiring/README.md)。该次独立安装验证使用已有依赖，导入/配方解析不初始化CUDA、不覆盖调用者精度；该安装产物的 GPU 前向和无预装脑影像软件的干净 Conda 整例仍未验证。cbaffa GPU 整例见上方独立报告。

[当前查询边复用报告](../../validation/recon_all/optimizations/20261011_topology_query_half/README.md)另存完整首缺陷、第四候选及直接库入口的各自冻结源码、全部负结果和脑图；没有将旧整例版本改标为新扫描核。[九缺陷完成报告](../../validation/recon_all/optimizations/20261011_topology_genetic_repair_cache/README.md)保留旧基线对照、原生重复和双向点到三角面距离：最大值约1.090/6.828 mm，整体指标等效未判定。

安装复核复用冻结源码及其逐文件清单；以下函数只构建、安装和检查入口，不运行影像算法：

```python
from pathlib import Path
from verify_recon_genetic_profile_install import verify  # tools目录须在PYTHONPATH中

installation_report = verify(
    source_directory=Path("/data/frozen-source"),  # 含pyproject、README、licenses、src的冻结包源码
    source_manifest=Path("/data/SOURCE_MANIFEST.json"),  # files逐项包含相对路径、bytes和sha256
    output_directory=Path("/data/new-install-check"),  # 尚不存在且位于冻结源码之外
    python_executable=Path("/conda/envs/fnit/bin/python"),  # 已安装主页依赖、setuptools及wheel的解释器
)
```

四项参数均为必填 `Path`，没有隐式默认目录。输出目录包含构建副本、wheel、独立安装目标、入口报告和总 `report.json`；返回字典记录代码/程序哈希、完整构建及安装墙钟、fresh CLI时间、导入前后精度设置。入口核对清单中完整EM、可选GPU距离、动态重连、有序扫描、程序边界、MNI、统计缓存与隔离worker等关键模块在源码、wheel及安装目标中的SHA，确认库默认Python扫描和具名配方选项；另外逐字节核验pial放置、清理、结构质量、观测器和显存采样五模块，并从实际安装入口读取cleanup_constraint_mode默认source_rip。此次检查不执行清理或GPU前向，不能代替同输入回归或网格验收；不调用扫描核或影像计算。它另复用所选 N4 与完整Python配方的生产依赖预检，报告 `n4_runtime_dependencies` 中的 SimpleITK/ITK 版本和扩展库 SHA、`python_runtime_dependencies` 中的依赖状态，以及实际安装的 `fnit-verify-recon-all-python` console 返回值。四个worker的fresh CLI help显式继承生产禁令，只核验解析和导入，不运行影像。NVML保持可选，资产不读取，CUDA不初始化；不运行 N4、不下载依赖、不改变精度或线程。原冻结源码不改写；清单路径越界、大小/SHA不同、已有输出、缺少依赖或安装/入口失败会抛异常，详细日志只保存在本次目录。命令行同名参数为 `--source-directory`、`--source-manifest`、`--output-directory`、`--python-executable`。本工具属于FNIT包装验证，没有原软件等价命令，也没有影像或坐标输出。

当前配方增加输入DAG选项后共有22个键，旧安装工具曾硬编码21并拒绝合法的新配方。安装检查现按具名后端与依赖选项校验，不把字典键数当作阶段数或算法完整性。原16模块核验见[首轮集成安装回归](../../validation/recon_all/optimizations/20261011_integrated_python_install/README.md)；完整EM合并后的24模块、实际console与依赖检查见[后续安装回归](../../validation/recon_all/optimizations/20261011_integrated_python_install_v2/README.md)；四个隔离worker守卫合并后的29模块与实际入口见[最新冻结安装回归](../../validation/recon_all/optimizations/20261011_integrated_python_install_v3/README.md)。各报告绑定各自实测提交，不将安装检查解释为整例完成。

生产入口和整例 benchmark 在配置分配器、加载模型、创建输出和启动子进程之前检查所选 N4 的依赖。`simpleitk` 要求主页安装已声明的固定版本；缺失或损坏时直接抛异常，不能退回其他算法。原始 T1 首次失败、修复后的本地合同与安装收据分别见[依赖接线与实际启动报告](../../validation/recon_all/optimizations/20261011_python_n4_dependency_route/README.md)。这次失败的 GPU 进程归属尚未解析，报告保留设备占用和进程合计观察值，不把它们称作 FNIT 整例峰值。

当前有序扫描接线的完整源码清单及新安装结果见[配方安装回归](../../validation/recon_all/optimizations/20261011_python_ga_blocker_route/README.md)。另一个冻结安装产物的原始T1 GA运行继续使用旧Python扫描；其[前段实际回归](../../validation/recon_all/optimizations/20261011_python_ga_volume_prefix/README.md)验证14张MRI数组、几何、dtype和LTA与旧Python前段相同，没有新增前段数值退化。它不被改标为新Numba配方的整例；已有N4对control差异仍单列。

## 6. 更新和 benchmark 记录

新版已安装 NumBa 配方也实际从原始 T1 连续到 filled：[新旧前段回归](../../validation/recon_all/optimizations/20261011_python_ga_raw_v3_prefix/README.md)的14张MRI数组、几何、dtype、标签和LTA相同。六个MGZ文件SHA差异仅来自各自路径的变换标签或fill来源记录，未改写生产文件，文件逐字节一致仍为8/14。两份整例仍在拓扑阶段，该报告不包含后续查询边复用核、white/pial或最终脑区指标的整例结果。

- 2026-10-11：公开单例、batch、CLI 与 worker 贯通显式 `python-ga`，原默认和数字链保留。
- 传递实际半球线程，避免双侧 GA 各自默认四线程；发布前要求真实完成与最终质量门，不重复 remesh。
- 修正新增 GA 状态目录的发布规则，保留目录层级、报告 SHA 和同名文件，拒绝跨半球写入。
- 完整 Python GA 配方启用已回归的保留面索引缓存，绑定实际参数及完成报告；安装产物包含当前库入口，首版收据仍按原版本保留。
- 修正整例benchmark入口的GA参数支持及配方透传；真实main参数回归在缺失T1预检处停止，不执行夹具影像。
- 原始 T1 启动暴露缺少已声明 N4 依赖；提前验证生产/包装入口，并扩展实际 wheel 入口检查，保留失败运行和修复验证的各自版本。
- 完整首缺陷和实际库入口回归通过后，GA配方显式启用完整Numba阻挡扫描；实际后端写入参数和发布收据，缺失或改变即失败。冻结运行继续使用各自源码，不原地改写为新版本。
- 当前查询边向量及叉积只在一次扫描内复用，近零和退化情况继续调用原完整谓词；完整首缺陷、第四候选及直接库入口回归分开记录，生产算法和候选域不变。
- 完整缓存搜索的科学运行继续使用各自带哈希冻结源码；文件适配器不使用 benchmark 的全局猴补，不读取官方结果。

## 7. 参考文献与原代码

- [FNIT recon-all 源码](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/tree/main/src/fnit/recon_all)。
- [固定 FreeSurfer 源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)、[mris_fix_topology](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a/mris_fix_topology)。
- Fischl B et al. Automated manifold surgery: constructing geometrically accurate and topologically correct models of the human cerebral cortex. IEEE Transactions on Medical Imaging, 2001, 20:70–80. [DOI](https://doi.org/10.1109/42.906426)。
