# 单 T1 N4 的独立 Python 库实现

## 1. 功能与范围

`n4_simpleitk` 用 SimpleITK 的完整 N4 Python API 替代原版阶段程序。它执行缩小、四层偏置拟合、全分辨率 B-spline 重建、指数与除法，再复用 FNIT 的量化和 nibabel 文件写出；不执行 FreeSurfer 命令、不读取参考结果，也不是旧平滑残差近似。

用户允许通用 Python 计算库的编译内核。这里使用 SimpleITK 2.5.6 / ITK 5.4，不是从 FreeSurfer 安装目录复制程序。完整 Torch N4 和独立原生 benchmark 仍保留。本页底层兼容函数默认保留native，显式 `n4_backend="simpleitk"` 选择此 CPU 实现；整例profile由主调度另行选择和验证。

```mermaid
flowchart LR
    A[自产orig 三维XYZ网格] --> B[shrink4 全1掩膜]
    B --> C[完整4层×50轮N4 拟合1线程]
    C --> D[全分辨率log偏置场]
    D --> E[Exp / Divide FP32输出]
    E --> F[clip与floor加0.5]
    F --> G[同网格uint8 nu0]
    G --> H[既有make_nu包装器]
```

## 2. Python 调用、全部输入和输出

```python
from fnit.recon_all.input_n4_chain import run_n4_stage

n4_report = run_n4_stage(
    input_path="mri/orig.mgz",            # 自产三维conformed T1，不传参考图
    output_path="mri/tmp/nu0.mgz",        # 同网格uint8新输出
    n4_backend="simpleitk",              # 完整Python库实现，不调用原版程序
    n4_execution="in-process",           # 此CPU后端不使用Torch isolated worker
    native_binary=None,                   # 无需原生程序；SimpleITK分支忽略此项
    device="cpu",                        # N4实际为CPU；不会初始化CUDA
    threads=4,                           # 拟合固定1，全分辨率重建使用4线程
    profile=False,                       # 不增加Torch子段同步
    report_path="scripts/n4-python.json", # 可选JSON，记录实际库版本和SHA
)
```

| 参数 | 类型、默认值和含义 |
| --- | --- |
| `input_path` | 必填 `str/Path`，三维 MGH/MGZ；有限、非负、含正强度，各轴至少8体素；强度无物理单位 |
| `output_path` | 必填，新三维 MGH/MGZ路径；不重采样、不改变 XYZ 顺序、shape 或毫米 affine |
| `n4_backend` | 默认 `native`；可选 `native`、完整 `torch` 或完整 `simpleitk`；本页针对后者 |
| `n4_execution` | 默认 `in-process`；`isolated` 仅适用于既有 Torch/cuda:N 后端，SimpleITK组合直接报错 |
| `native_binary` | 默认 None；native必填，SimpleITK不使用，不发现或执行原版程序 |
| `device` | 默认 cpu；SimpleITK实际CPU，与调用者Torch当前设备和精度无关 |
| `threads` | 默认4，正整数且不接受bool；本后端仅控制全分辨率重建，拟合固定单线程 |
| `profile` | 默认False；SimpleITK自身返回分步墙钟，不依赖GPU同步 |
| `report_path` | 默认None；指定则保存算法、依赖版本/SHA、文件SHA和时间等JSON；读写错误传播 |

返回字典包括 `backend="simpleitk_python"`、`execution`、`device="cpu"`、`used_native_programs=False`、`simpleitk_version`、`itk_version`、`library_sha256`、`recipe`、`internal_geometry`、`fitting_threads`、`reconstruction_threads`、`current_level`、`elapsed_last_level_iterations`、线程恢复值、分步秒数、`array_stage_seconds`、`total_seconds`、文件SHA及 `output`。完整阶段报告的文件墙钟包含读取与写出，benchmark另在调用外记录包括元数据处理的墙钟。

直接文件API是 `n4_simpleitk.correct_volume(input_file, output_file, *, reconstruction_threads=1, fitting_threads=1, profile_path=None)`；`fitting_threads` 是新增低层显式选项，正整数且不接受bool，默认仍1。兼容路由 `n4_itk.correct_volume(..., backend="simpleitk", binary=None)` 和上面的 `run_n4_stage` 继续采用拟合1线程，`threads` 只控制重建；没有把新拟合选项自动传播到生产。前两个路径必填，返回同一格式报告；默认native兼容路由仍需binary。

数组API是 `correct_array(image, *, spacing, reconstruction_threads=1, fitting_threads=1)`：`image` 为 XYZ 三维 NumPy float32强度，`spacing` 为三个有限正数、单位毫米；返回同shape新float32数组与实际报告，不修改输入。内部坐标设 origin=0、identity direction，沿XYZ轴按原图间距计算，匹配既有 raw N4配方；文件输出保留原scanner-RAS affine，不把内部坐标写入输出头。新增报告 `actual_fitting_work_units/actual_fitting_filter_threads` 保存滤波器实际设置，不只记录请求值。

```python
from fnit.recon_all.n4_simpleitk import correct_volume

file_n4_report = correct_volume(
    input_file="mri/orig.mgz",                  # 自产conformed T1，三维MGH/MGZ
    output_file="diagnostic/nu0.mgz",            # 新输出路径，同网格uint8
    fitting_threads=1,                          # 默认单线程拟合；多线程需完整同输入回归
    reconstruction_threads=4,                   # 全分辨率重建线程，与拟合独立
    profile_path="diagnostic/n4-profile.json",   # 实际参数、分段时间与线程恢复状态
)
```

固定配方：全1 uchar掩膜，maskLabel1，shrink=(4,4,4)，splineOrder3，初始控制点(4,4,4)，iterations=(50,50,50,50)，convergence0，histogram200，FWHM0.15，WienerNoise0.01。成功后滤波器的末层计数可能为51，报告保留库实际返回值，不将它改写成总200轮计数。

本模块锁保护自己的SimpleITK全局线程设定，并在成功或异常后恢复原值；与其他库调用并发时应使用独立进程。不修改父CUDA当前设备、分配缓存或TF32，不启用半精度。非法输入/参数抛ValueError，缺依赖抛ImportError，N4/读写异常传播；不降级到近似结果。

运行前可显式调用 `require_n4_runtime_dependencies(*, n4_backend)`：SimpleITK 后端检查当前解释器实际加载的固定版本 `2.5.6`、必要 N4 API 和可读计算库，返回 `backend/required/simpleitk_version/itk_version/library_sha256`。native/Torch 路由不导入 SimpleITK，返回的 `required=False` 仅表示无需本依赖，不能代替它们的程序或设备校验。纯参数函数 `validate_n4_execution` 不检查安装；CLI help 或参数解析不等于依赖可用。

```python
from fnit.recon_all.input_n4_chain import require_n4_runtime_dependencies

dependency_report = require_n4_runtime_dependencies(
    n4_backend="simpleitk",  # 所选计算后端；不会读取T1、创建subject或启动模型
)
print(dependency_report["simpleitk_version"], dependency_report["library_sha256"])
```

`run_input_n4_chain` 在输入 Talairach 子流程之前、`run_n4_stage` 在 N4 文件分派之前执行该预检；直接 `n4_simpleitk.correct_volume` 也在 nibabel 读入之前保护。包不存在、ABI 导入失败、不是已验证的 2.5.6、必要 API 缺失或库文件不可读均抛明确 `ImportError`，不创建新输出、不换后端。预检只加载并核查库，不建立图像或滤波器、不改 ITK 线程、CUDA 或 TF32；SHA 按实际平台库记录，不要求所有平台采用同一 wheel SHA。

## 3. 命令行与安装

主页 Conda/Python 依赖应安装固定 `SimpleITK==2.5.6`。本次实际wheel是 cp311-abi3 manylinux2014 x86_64，52,798,369字节，SHA-256为 `99242c333ed17138134e9749f3a484518f41baebab0fc0f0fe7c657ab8090c07`；其他平台的wheel哈希不同，应分别校验。

已有 wheel 的独立 target 安装可使用以下本地目录变量；这不会修改现有环境。target 必须加入实际运行解释器的 `PYTHONPATH`，不能只把 wheel 或依赖声明放在源码旁边。

```bash
# simpleitk_wheel 指向已下载并验证SHA的2.5.6 wheel；dependency_target 是新独立目录
simpleitk_wheel=/path/to/simpleitk-2.5.6-cp311-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
dependency_target=/path/to/new-fnit-dependencies
python -m pip install --no-index --no-deps --target "$dependency_target" "$simpleitk_wheel"
export PYTHONPATH="$dependency_target${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from fnit.recon_all.input_n4_chain import require_n4_runtime_dependencies; print(require_n4_runtime_dependencies(n4_backend="simpleitk"))'
```

```bash
# CPU完整N4，i/o必填；重建默认1，这里显式设置4
python -m fnit.recon_all.n4_simpleitk \
  --i mri/orig.mgz \
  --o mri/tmp/nu0.mgz \
  --fitting-threads 1 \
  --reconstruction-threads 4 \
  --profile scripts/n4-python.json

# 同一个兼容文件接口，SimpleITK分支无需binary
python -m fnit.recon_all.n4_itk \
  --backend simpleitk \
  --i mri/orig.mgz \
  --o mri/tmp/nu0.mgz \
  --reconstruction-threads 4
```

`--profile` 可选，默认不写单独JSON；失败退出非零。低层 `n4_simpleitk` 的 `--fitting-threads` 默认1，仅控制拟合；兼容 `n4_itk` CLI 不新增这一选项。原始输入链的 `python -m fnit.recon_all.input_n4_chain` 也接受 `--n4-backend simpleitk`，其他T1、subject、权重和资产等必填参数保持原有接口；完整输入说明见该模块docstring。N4 本身无需外置权重、atlas或许可证，原生程序只用于独立验证。

## 4. 对应原软件和固定参数

标准参考是 recon-all 的 N4偏置阶段及后续 `make_nu` 强度包装，不是把 `mri_nu_correct.mni` 当作此配方的独立等价命令。仓库基准程序 `fnit_n4_itk` 以固定 ITK 5.4.7 配方源码在 Conda 构建；它采用raw XYZ float32字节输入/输出，固定配方见上文。其实际SHA与源码SHA在本次报告中记录，生产SimpleITK不调用它。N4的Shrink/GetLogBiasFieldAsImage/Exp/Divide为库API步骤，没有虚构单独FreeSurfer CLI。

原始整例官方对照命令仍为独立环境中的 `recon-all -i public_T1w.nii.gz -s reference_subject -all`；本次仅运行冻结同输入N4，不将它写成新的官方整例或FNIT整例。

## 5. 两例真实精度、时间和脑图

### 当前依赖预检修复

2026-10-11 的新原始 T1 运行在 Talairach 后因缺少 SimpleITK 停止；此前 CLI/包导入检查未验证所选 N4 的计算库。这是成熟 N4 子函数的失败时机问题。已增加运行依赖预检，N4 参数和数值计算不变；本轮复用已声明 wheel 在新私有 target 安装，不修改旧 Conda prefix 或失败运行。当前完整同输入 N4 回归、安装 SHA 和合同证据见[依赖预检报告](../../validation/recon_all/optimizations/20261011_n4_dependency_preflight/README.md)。该报告仅核依赖恢复和 N4 阶段，不把旧失败改为整例成功，也不重新标注下列历史 benchmark。

### 最新：完整文件阶段的拟合线程1 / 4

本轮实际模块SHA为 `c14e8eb46fb408926ddb9a7fe3d35d97bb5201382928e87226f71f8e01041b90`，冻结时Git `86ae36bde8c0ec5960fbb9142af2b060c24d1d25` 加独立overlay；同两例自产orig、同主机CPU100–103、总4线程，各完成 `[1,4,4,1]` 完整文件API。重建始终4，默认拟合仍1，生产兼容路由未切换。

| 完整文件API ABBA结果 | sub-07 | sub-06 |
| --- | ---: | ---: |
| 拟合1→4中位墙钟，秒 | 184.594880→52.399055 | 183.716236→51.533518 |
| 本组阶段观察缩短 | 71.6140% | 71.9494% |
| 线程1 vs4：校正float最大 / P99 | 0.0029907227 / 0.0017013550 | 0.0029296875 / 0.0019226074 |
| 线程1 vs4：nu0不同体素 / 最大 | 2406 / 1 | 3233 / 1 |
| 线程4相对既有原生：nu0不同体素 / 最大 | 2420 / 1 | 3134 / 1 |
| 各线程设置自身两次log场、float、nu0差异 | 全0 | 全0 |

原约179–180秒的拟合降到约47–48秒，其余完整算法及IO调用保持。实际filter threads/work units均匹配请求值，八次几何、dtype和线程恢复通过；七项adapter合同1.733秒通过。跨线程变化可重复，尚未定位某个库内部归约，不能归为随机性。严格逐位复现未通过，整体指标等效未判定；正在另项验证原始T1→filled传播，不能把本阶段71%缩短当整例提速。完整读写、导入、子进程墙钟、误差脑图、1,496数值叶及失败诊断见[本轮线程报告](../../validation/recon_all/optimizations/20261010_n4_fitting_threads/README.md)。

### 首次独立适配器的冻结同输入验证

公开 ds000114 sub-07/sub-06，输入为 `79a41cdd` 自产完整链的 `orig.mgz`。候选基线 `54f681dcab210a22c2e6862b2854a7033b5a208d` 加独立模块SHA；同A100节点CPU24–27、四线程预算，拟合1、重建4。库实现是CPU，不占用GPU或声称GPU显存为0。

| 冻结同输入差异 | sub-07 | sub-06 |
| --- | ---: | ---: |
| FP32不同体素 | 6,416,123 | 8,038,768 |
| FP32最大绝对误差 | 0.00006103515625 | 0.0001068115234375 |
| FP32 P99 | 0.000034332275390625 | 0.0000457763671875 |
| FP32平均绝对误差 | 0.00000207115618 | 0.00000537340502 |
| 量化uint8不同体素 | 28 | 99 |
| uint8最大 / P99 | 1 / 0 | 1 / 0 |
| 原生重复FP32 / uint8差异 | 0 / 0 | 0 / 0 |

两次原生输出完全重现，因此不能把候选差异归为随机性。SimpleITK与独立ITK源码版本不同，但本次未进一步证明哪项内部运算造成尾差；真实浮点误差跨越少量量化边界。严格复现未通过，后续自产归一化/注册及网格传播、最终指标等效要另外验证，不能因为均值小就宣布无影响。

| 完整数组/原生调用，秒 | sub-07 | sub-06 |
| --- | ---: | ---: |
| native第一次（含raw IO） | 163.2296925 | 164.9746436 |
| Python完整数组调用（含返回元数据） | 181.9413680 | 181.1083046 |
| Python完整MGZ文件API（含加载/写出） | 186.3816829 | 185.2901246 |

Python数组主耗时分别为拟合180.697/180.031秒，重建0.249/0.354秒，Exp/Divide/下载0.411/0.206秒；该实现解决独立调用边界，本次没有测到提速。完整MGZ文件接口两例相对此前Python数组量化结果均0体素差异，原affine、shape和uint8一致，量化相对native仍为28/99而非归零；实际路由及wrapper SHA另行绑定，见[当前机器可读报告](../../validation/recon_all/optimizations/20261010_n4_simpleitk/README.md)。七项合同通过，包含非等距文件坐标、最近整数写出、异常线程恢复与不进入Torch worker。父进程CUDA状态和TF32不被触碰。这里没有新原始T1整例、独立部署或最终20GB整例证据。

## 6. 更新与 benchmark 记录

- 2026-10-11：修复成熟 SimpleITK 子函数直到读入 T1/执行上游后才发现缺包的行为，新增独立运行依赖预检，保留纯参数校验。缺包、错误版本、API/库不可用明确早报；N4 fitting/reconstruction 规则、默认值和线程恢复不变。本次独立安装仅复用已有 wheel，整例重新验证由主调度另行冻结。
- 2026-10-10：低层数组/文件/CLI新增显式 `fitting_threads=1`，实际请求与filter threads/work units写入报告。默认1及兼容路由保持；两例完整ABBA实测提速、重复性与新增量化差异发布，不提前判定整体等效。
- 2026-10-10：增加完整SimpleITK Python适配器、既有文件兼容路由和实际input_n4_chain路由；保留原native/torch候选及默认。完整两例数组对照与原生重复检验已完成，浮点和量化尾差逐项发布；完整文件API在同输入另测。
- 同日修复成熟input chain只允许native/torch的后端校验：simpleitk明确CPU、in-process，不会误入Torch isolated worker；线程恢复及路由合同通过，原参数含义不变。
- 较早完整Torch N4自产连续链的放大、注册/网格问题仍由[四输入归因](../../validation/recon_all/optimizations/20261010_n4_gca_cross_inputs/README.md)记录；不将该Torch结果重标为本实现结果，也不删除仍承担诊断作用的实现。

## 7. 原实现与参考文献

- [FNIT独立N4 Python实现](../../src/fnit/recon_all/n4_simpleitk.py)
- [SimpleITK完整N4与全分辨率重建示例](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html)
- [SimpleITK N4参数接口](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1N4BiasFieldCorrectionImageFilter.html)
- [SimpleITK源码](https://github.com/SimpleITK/SimpleITK)、[ITK N4源码](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.0/Modules/Filtering/BiasCorrection/include/itkN4BiasFieldCorrectionImageFilter.hxx)
- Tustison NJ et al. N4ITK: improved N3 bias correction. IEEE Trans Med Imaging. 2010;29:1310–1320. doi:10.1109/TMI.2010.2046908.
