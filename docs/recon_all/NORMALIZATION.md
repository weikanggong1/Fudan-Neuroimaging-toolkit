# T1 强度归一化的 Python/PyTorch 实现

第一轮函数对应 FreeSurfer 8.2 的 `mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz`：读取 conform 后的 `nu.mgz` 和配套 `talairach.xfm`，依次执行一维样条、温和校正及两轮三维控制点/偏置场计算，输出 `T1.mgz`。当前 `fnit-recon-all` 已调用这个 Python 阶段；后续完整皮层指标仍未通过整例验收。函数不启动 FreeSurfer 程序，也不需要模型权重或额外模板。

从仓库根目录运行 `python -m pip install -e '.[recon-all-python-stages]'` 安装本阶段及 Numba 等依赖。此阶段使用 PyTorch、NumPy、SciPy、Numba 和 NiBabel；精确 CPU 高斯路径依赖 Numba。独立的 [N4 阶段](N4_ITK_CONDA.md)使用仓库内 Conda C++/ITK，本归一化函数不调用它。

## 命令行与 Python 调用

```bash
fnit-normalize \
  --input /subjects/sub01/mri/nu.mgz \
  --xfm /subjects/sub01/mri/transforms/talairach.xfm \
  --output /subjects/sub01/mri/T1.mgz \
  --device cuda:0
```

`--input` 指 conform 后的 `nu.mgz`，`--xfm` 指其 Talairach 变换，`--output` 指要写入的 uint8 `T1.mgz`，`--device` 指计算设备。函数返回的字典还含 `device`、`three_d_iterations`、`peaks`、`controls`、`propagation`、`smoothing` 和 `three_d_passes`，用于复核各轮输入选择与耗时。

无 CUDA 时设 `--device cpu`；省略该项则可用 CUDA 时选 CUDA，否则选 CPU。命令打印含各步耗时的 JSON。`--three-d-iterations 0` 和 `1` 对应 FreeSurfer 中间检查点 `-n 0`、`-n 1`，默认为 `2`。`--diagnostic-dir /path` 将各三维迭代的 float32 输入、控制点掩膜和 float32 偏置场写成 MGH 文件。

```python
from fnit.recon_all.normalization import normalize_t1

report = normalize_t1(
    input_file="/subjects/sub01/mri/nu.mgz",  # conform 后的强度图
    xfm_file="/subjects/sub01/mri/transforms/talairach.xfm",  # 配套 Talairach 变换
    output_file="/subjects/sub01/mri/T1.mgz",  # uint8 输出图路径
    device="cuda:0",  # PyTorch 运算设备
)
# report["total_seconds"] 是函数墙钟秒数，report["steps"] 是各步耗时字典。
print(report["total_seconds"], report["steps"])
```

输出是 uint8 `T1.mgz`；各轮之间保留 float32 图像。输入 `nu.mgz` 必须是 FreeSurfer conform 网格，Talairach 变换需与其几何信息配套。实现参照 FreeSurfer 提交 `d932c45b7941662ea380a05efef580568b98d41a`。在 `fs_sub01` 的 CPU 对照中，中间 float32 图、控制点掩膜、偏置场及最终 256³ uint8 图逐体素一致；H100 对照的最终 `T1.mgz` 也一致。[检查点和计时报告](../../validation/recon_all/python_gpu_port/experimental/NORMALIZE_FIRST_PASS.md)记录了逐步结果。现有验证只覆盖一例。

## CPU 与 GPU 分工

| 设置 `--device cuda:0` 时的步骤 | 执行位置 |
| --- | --- |
| MGH/MGZ 读写、Talairach 解析、一维直方图和样条系数 | CPU，NiBabel/NumPy/SciPy |
| 一维体素缩放及各轮偏置场应用 | H100，PyTorch float32 |
| 温和校正、两轮三维控制点选择、组织直方图和邻域判断 | CPU，NumPy/SciPy；有序离群清理使用共享 Numba 内核 |
| Voronoi chessboard 距离和索引排序 | CPU，SciPy/NumPy |
| Voronoi 波前平均与三次 sigma-8 高斯卷积 | H100，PyTorch float32 |

精确 CPU 路径以 Numba 按源码顺序累加高斯值。生产函数没有 subprocess 调用或 FreeSurfer 程序查找；官方程序只用于单独生成对照输出和计时。

相同 `fs_sub01` T1、同主机且不写诊断文件的 CLI 配对观察：headcw CPU 官方 **76.94 秒**、Python **60.55 秒**；gpucw1 H100 GPU1 官方 **116.31 秒**、Python **75.60 秒**。两次 Python 输出各有 **0/16,777,216** 个差异体素。H100 一次 68.87 秒的常驻 Python 运行中，两轮 CPU 控制点搜索耗时 20.24 和 32.60 秒，是目前主要时间开销。分步命令、哈希和跨被试待验收项见[验证报告](../../validation/recon_all/python_gpu_port/experimental/NORMALIZE_FIRST_PASS.md)。

## 带 aseg 与 brainmask 的第二轮归一化

第二个函数对应 `mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz`。输入为同一 1 mm、256³ conform 网格的 `norm.mgz`（uint8 强度）、`aseg.presurf.mgz`（整数标签）、`brainmask.mgz`（uint8 掩膜），输出同网格 uint8 `brain.mgz`；也返回步骤和耗时报告。Fast Marching 内侧白质 ridge、离群点过滤、初始偏置场、温和校正和两轮三维迭代由 NumPy/Numba/PyTorch 完成，不调用 FreeSurfer。调度使用用户选择的 CPU/CUDA 设备；控制点搜索仍在 CPU，偏置场计算使用选定设备。

```bash
fnit-normalize-aseg \
  --norm /subjects/sub01/mri/norm.mgz \
  --aseg /subjects/sub01/mri/aseg.presurf.mgz \
  --brainmask /subjects/sub01/mri/brainmask.mgz \
  --output /subjects/sub01/mri/brain.mgz \
  --device cuda:0
```

`--norm`、`--aseg`、`--brainmask` 依次指定三张同网格输入图；`--output` 指输出 `brain.mgz`，`--device` 选 CPU/CUDA。Python 函数的 `device=None` 默认有 CUDA 时使用 `cuda:0`，否则使用 CPU；`three_d_iterations=2`，只接受 0、1、2。同网格检查或迭代次数检查失败会抛出异常。返回字典除 `total_seconds` 外，还记录 ridge 和初始偏置场耗时、控制点数量、白质峰值及后续迭代信息。

```python
from fnit.recon_all.normalization import normalize_t1_aseg

report = normalize_t1_aseg(
    norm_file="/subjects/sub01/mri/norm.mgz",  # GCA 归一化强度图
    aseg_file="/subjects/sub01/mri/aseg.presurf.mgz",  # 皮层下结构标签
    brainmask_file="/subjects/sub01/mri/brainmask.mgz",  # 脑掩膜
    output_file="/subjects/sub01/mri/brain.mgz",  # 归一化脑图输出路径
    device="cuda:0",  # 设备；两例真实 T1 的同输入输出与 CPU 逐体素一致
    three_d_iterations=2,  # 三维控制点与偏置场重复两轮
)
# report["total_seconds"] 是墙钟秒数，report["completion"] 记录后续归一化步骤。
```

在冻结的 `fs_sub01` 输入上，独立 Python ridge、过滤后的控制点掩膜、离群图和初始 float32 偏置场均与官方诊断图一致。默认两轮的完整 `brain.mgz` 为 **0/16,777,216** 个差异体素，MGH 头前 284 字节和体素负载一致。新运行的官方文件在末尾多 996 字节元数据，旧存档输出多 451 字节，因此完整文件哈希不同。这只是单例阶段验证，整例数值一致性仍需检验。[第二轮报告](../../validation/recon_all/python_gpu_port/NORMALIZE_SECOND_PASS.md)列出原始证据。

2026-09-30 在两例新生成的 FNIT 连续链输入上，CPU/CUDA 的 `brain.mgz` 各有 **0/16,777,216** 个差异体素，仿射矩阵一致。`sub-01` 同在 gpucw1 上 CPU 阶段 85.17 秒、CUDA 阶段 59.15 秒；`sub-02` 的 CPU 在 nodecw10 为 85.96 秒，CUDA 在 gpucw1 为 71.09 秒，跨主机时间不能算配对加速。两次 CUDA 父子进程显存每 2 秒采样峰值均为 1,247,805,440 字节；这不是整例显存。输入、源码、计时与采样见[本轮性能记录](../../validation/recon_all/python_gpu_port/performance_20260930/README.md)。

## 控制点的有序离群清理

`gentle_controls` 和 `controls_3d` 原先各有一个 Python 逐控制点循环。现在两者复用
`_remove_outliers_ordered`：串行 Numba 编译，并缓存编译产物。邻域判断仍是当前点
周围裁剪的 3×3×3 体素，邻居少于两个则立刻删除；扫描仍按 z、y、x 顺序。
后面的点读取已经更新的控制图，因此不能换成一次卷积后同时删除，也不能并行
更新。这一步只计数 bool 值，没有浮点累加、fastmath 或精度设置变更。
CPU 内核依赖现有主页 `environment.yml` 的 Numba，不新增安装依赖。

| 函数与参数 | 输入、默认值与限制 | 输出 |
| --- | --- | --- |
| `gentle_controls(image)` | 三维 PyTorch 强度张量；在输入网格以 float32 截断强度后搜索固定 7/5 体素窗口；没有额外算法参数 | 同 shape、同设备 uint8 0/1 控制图；dict 记录 `first_7x7x7`、`after_5x5x5`、`after_outlier_removal` 数量 |
| `controls_3d(source, wm_peak=None, gm_peak=None)` | 三维 NumPy 强度数组；内部为 float32，窗口强度沿用 int16 转换；峰值单位同强度，任一 None 时重估两者 | 同 shape uint8 0/1 控制图；dict 记录组织峰、两组锚点、两次扩展及清理前后数量 |
| `_remove_outliers_ordered(control)` | 内部三维 bool 控制图，按 (x,y,z) 索引，原地更新；只由上述函数调用 | 删除点数（整数）；输入数组被更新 |

三个函数都不改变仿射、坐标空间或体素大小；邻域单位是体素，输出标签仅为
控制点含义，不是组织分割。公开函数遇到非三维输入抛出 `ValueError`；图像没有
可用组织区域时，保留现有组织峰或区域搜索的异常。Numba 缓存目录不可写等安装
问题不被静默替换成近似实现。这些函数属于前述 `mri_normalize` 命令的内部步骤，
没有独立官方 CLI。

```python
import nibabel as nib
import numpy as np
import torch
from fnit.recon_all.normalization.normalize_gentle_source import gentle_controls
from fnit.recon_all.normalization.normalize_3d_controls import controls_3d

input_image = nib.load("/subjects/sub01/mri/nu.mgz")  # 同一 conform 网格的输入图
input_values = np.asarray(input_image.dataobj, dtype=np.float32)  # 归一化强度单位
input_tensor = torch.as_tensor(input_values, device="cuda:0")  # 输出控制图返回此设备
gentle_control, gentle_report = gentle_controls(image=input_tensor)  # 固定窗口的温和控制点
three_d_control, three_d_report = controls_3d(
    source=input_values,  # (x,y,z) 强度数组；真实流程应传入本轮校正后的强度图
    wm_peak=None,  # 自动估计白质强度峰
    gm_peak=None,  # 自动估计灰质强度峰；任一峰未提供时同时重估
)
nib.save(
    nib.MGHImage(three_d_control, input_image.affine),  # 同网格 uint8 0/1 控制图
    "/diagnostic/sub01/three_d_control.mgh",  # 独立诊断路径；父目录须存在
)
```

同输入回归使用 [配对脚本](../../validation/recon_all/python_gpu_port/benchmark_normalize_ordered_outliers.py)。
它同时加载冻结旧版的 `normalize_gentle_source.py`、`normalize_3d_controls.py`，
只替换这两个算子，其余阶段使用候选源码，报告明确记录这一范围。两轮性能运行
按旧/新、新/旧次序执行，默认不预热，记录首次编译或缓存加载前后的 signatures；
计时含输入校验、读写、传输，CUDA 开始和结束均同步指定设备。另做独立诊断运行，
保存温和控制图、每轮三维输入/控制图、第二次归一化的 ridge/过滤/移除控制图、
最终 T1/brain，并逐张比较不同值数、最大/P99 误差、dtype 和仿射。诊断写图耗时
单列，不当作生产阶段提速。报告保留输入、全部归一化源码、冻结旧模块、脚本和
输出 SHA-256；可另外传入官方 T1/brain 进行对照。

```bash
# 下列变量需指向已经验证的真实输入与冻结源码，不下载或读取新的官方产物。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python validation/recon_all/python_gpu_port/benchmark_normalize_ordered_outliers.py \
  --nu "${SUBJECT_DIR}/mri/nu.mgz" \
  --xfm "${SUBJECT_DIR}/mri/transforms/talairach.xfm" \
  --norm "${SUBJECT_DIR}/mri/norm.mgz" \
  --aseg "${SUBJECT_DIR}/mri/aseg.presurf.mgz" \
  --brainmask "${SUBJECT_DIR}/mri/brainmask.mgz" \
  --baseline-controls "${BASELINE_CODE}/src/fnit/recon_all/normalization/normalize_3d_controls.py" \
  --baseline-gentle "${BASELINE_CODE}/src/fnit/recon_all/normalization/normalize_gentle_source.py" \
  --baseline-commit "${BASELINE_COMMIT}" \
  --code-commit "${CANDIDATE_COMMIT}" \
  --output-dir "${EMPTY_DIAGNOSTIC_DIR}" \
  --stage both --device cuda:0 --threads 4 --repeats 2 \
  --timing-context "同主机固定线程；另附 GPU 共享负载及进程显存采样"
```

`--nu`/`--xfm` 用于第一次，`--norm`/`--aseg`/`--brainmask` 用于第二次；`--stage`
默认 `both`，也可选 `first` 或 `second`。`--baseline-controls`/`--baseline-gentle`
和两个 commit 字段必须提供；源码哈希是实际算子绑定依据，commit 字段不能替代
哈希。`--device` 必须显式指定，`--threads=4`、`--repeats=2` 必须为正整数。
`--output-dir` 必须为空，否则报错；新文件均写入其中的分阶段子目录和 `report.json`。
`--reference-t1`、`--reference-brain` 可选，指向仅用于诊断的官方冻结输出。
`--timing-context` 为采样负载说明，默认未指定。PyTorch 显存统计显式指定设备；
缓存禁用时记为不可用，不写成零。父子进程同一时刻的总显存另由外部采样记录。

本次内核的真实数据回归与整例提速须由上述新报告判断，不能沿用前文的历史秒数。
局部边界测试保护扫描顺序和原地传播，不替代真实 T1 benchmark；最终整例仍须
从原始 T1 与新空目录运行，冻结归一化输入配对不构成连续整例。

## 参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. [doi:10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码提交](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。


### 本轮真实阶段配对（2026-10-01）

同一 sub-01 自产输入、gpucw1 CPU、4 线程，仅替换两处有序清理算子：第一轮归一化 84.645→80.945 s，第二轮 94.228→90.358 s。各计时含输入读取、运算和结果写出；诊断图另一次运行，未混入性能计时。最终影像和全部诊断控制图的体素、dtype、shape、仿射完全相同。基线源码为 1b8c36d，两处候选源码分别以 CPU 快照中的 SHA 绑定；[完整报告](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/normalize_sub01_pair.json)。sub-02 同输入 CPU 第一轮 84.114→80.687 s，第二轮 105.981→100.384 s；最终影像、全部诊断控制图、dtype、shape 和仿射同样一致，见 [sub-02 报告](../../validation/recon_all/python_gpu_port/performance_hotspots_20261001/normalize_sub02_pair.json)。候选生产代码已固定为 c24852054f3321c1142b1ae88fa3d2bf68329bb3；两例原始 T1 整例正在运行，此处 CPU 算子回归不证明 GPU 或整例提速。
