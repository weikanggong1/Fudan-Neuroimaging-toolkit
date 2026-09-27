# T1 强度归一化的 Python/PyTorch 实现

第一轮函数对应 FreeSurfer 8.2 的 `mri_normalize -g 1 -seed 1234 -mprage nu.mgz T1.mgz`：读取 conform 后的 `nu.mgz` 和配套 `talairach.xfm`，依次执行一维样条、温和校正及两轮三维控制点/偏置场计算，输出 `T1.mgz`。当前 `fnit-recon-all` 已调用这个 Python 阶段；后续完整皮层指标仍未通过整例验收。函数不启动 FreeSurfer 程序，也不需要模型权重或额外模板。

从仓库根目录运行 `python -m pip install -e '.[recon-all-python-stages]'` 安装本阶段及 Numba 等依赖。此阶段使用 PyTorch、NumPy、SciPy、Numba 和 NiBabel；精确 CPU 高斯路径依赖 Numba。独立的 [N4 SITK 阶段](N4_SITK_VALIDATION.md)使用 SimpleITK，本归一化函数不使用。

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
| 温和校正、两轮三维控制点选择、组织直方图和邻域判断 | CPU，NumPy/SciPy |
| Voronoi chessboard 距离和索引排序 | CPU，SciPy/NumPy |
| Voronoi 波前平均与三次 sigma-8 高斯卷积 | H100，PyTorch float32 |

精确 CPU 路径以 Numba 按源码顺序累加高斯值。生产函数没有 subprocess 调用或 FreeSurfer 程序查找；官方程序只用于单独生成对照输出和计时。

相同 `fs_sub01` T1、同主机且不写诊断文件的 CLI 配对观察：headcw CPU 官方 **76.94 秒**、Python **60.55 秒**；gpucw1 H100 GPU1 官方 **116.31 秒**、Python **75.60 秒**。两次 Python 输出各有 **0/16,777,216** 个差异体素。H100 一次 68.87 秒的常驻 Python 运行中，两轮 CPU 控制点搜索耗时 20.24 和 32.60 秒，是目前主要时间开销。分步命令、哈希和跨被试待验收项见[验证报告](../../validation/recon_all/python_gpu_port/experimental/NORMALIZE_FIRST_PASS.md)。

## 带 aseg 与 brainmask 的第二轮归一化

第二个函数对应 `mri_normalize -seed 1234 -mprage -aseg aseg.presurf.mgz -mask brainmask.mgz norm.mgz brain.mgz`。输入为同一 conform 网格的 `norm.mgz`、`aseg.presurf.mgz`、`brainmask.mgz`，输出 `brain.mgz`；也返回步骤和耗时报告。Fast Marching 内侧白质 ridge、离群点过滤、初始偏置场、温和校正和两轮三维迭代由 NumPy/Numba/PyTorch 完成，不调用 FreeSurfer。CPU 精确路径已对照，第二轮 CUDA 路径尚未与官方配对。

```bash
fnit-normalize-aseg \
  --norm /subjects/sub01/mri/norm.mgz \
  --aseg /subjects/sub01/mri/aseg.presurf.mgz \
  --brainmask /subjects/sub01/mri/brainmask.mgz \
  --output /subjects/sub01/mri/brain.mgz \
  --device cpu
```

`--norm`、`--aseg`、`--brainmask` 依次指定三张同网格输入图；`--output` 指输出 `brain.mgz`，`--device` 选 CPU/CUDA。返回字典除 `total_seconds` 外，还记录 ridge 和初始偏置场耗时、控制点数量、白质峰值及后续迭代信息。

```python
from fnit.recon_all.normalization import normalize_t1_aseg

report = normalize_t1_aseg(
    norm_file="/subjects/sub01/mri/norm.mgz",  # GCA 归一化强度图
    aseg_file="/subjects/sub01/mri/aseg.presurf.mgz",  # 皮层下结构标签
    brainmask_file="/subjects/sub01/mri/brainmask.mgz",  # 脑掩膜
    output_file="/subjects/sub01/mri/brain.mgz",  # 归一化脑图输出路径
    device="cpu",  # 已验证的精确 CPU 路径
)
# report["total_seconds"] 是墙钟秒数，report["completion"] 记录后续归一化步骤。
```

在冻结的 `fs_sub01` 输入上，独立 Python ridge、过滤后的控制点掩膜、离群图和初始 float32 偏置场均与官方诊断图一致。默认两轮的完整 `brain.mgz` 为 **0/16,777,216** 个差异体素，MGH 头前 284 字节和体素负载一致。新运行的官方文件在末尾多 996 字节元数据，旧存档输出多 451 字节，因此完整文件哈希不同。这只是单例阶段验证，整例数值一致性仍需检验。[第二轮报告](../../validation/recon_all/python_gpu_port/NORMALIZE_SECOND_PASS.md)列出原始证据。
