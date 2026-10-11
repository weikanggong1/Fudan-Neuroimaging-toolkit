# Pial 同网格输入预检修复（2026-10-11）

## 1．功能与本次问题

`place_pial_t1()` 原来读取 `brain.finalsurfs` 的空间变换，并按同一体素索引消费 WM/aseg，但没有预检两者是否错位。此次在强度准备、目标搜索和四轮优化前校验 MRI 头，并复用既有 `surface_ras_to_voxel()`，不增加重采样。

三张 MRI 必须是同 shape、同 scanner RAS affine 的三维网格；各头间距正且有限，white 坐标、完整体积几何和实际 surface RAS→voxel 变换有限。表面有效标记沿原格式允许 `1  # volume info valid`；`0`、缺失、无注释的多余文本或退化几何失败。不要求表面头等于 MRI 头，既有变换仍支持不同的有效表面几何。

MRI 各自的 `vox2ras_tkr` 仅检查有限性。WM/aseg 的体素索引未消费这两个矩阵；同一 scanner affine 的 MGH 方向/间距分解可以不同。实际输入检查发现了首版误用该条件，失败记录及哈希保留。

## 2．Python 调用、输入输出及参数

```python
from fnit.recon_all.place_pial_python import place_pial_t1

report = place_pial_t1(
    subject="/data/fnit/sub06",  # 含自产 MRI、white、两个标签和阈值表
    hemisphere="lh",  # 半球，只接受 lh/rh
    output="/data/diagnostic/lh.pial.T1",  # 完整成功后写出的独立表面
    max_steps=400,  # 四轮总迭代保护上限，不跳过轮次
    sampling_backend="cpu",  # 已验证强度采样；可选 torch/triton
    regularization_backend="cpu",  # 保留有序梯度；可显式选 torch
    candidate_backend="snapshot",  # 完整空间候选；GPU 模式可选 torch_snapshot
    candidate_grid_cells_per_axis=2,  # CPU 完整候选网格
    retained_mht_backend="compiled",  # Numba 有序碰撞重试
    cleanup_marking_backend="source_numba",  # 完整自有源规则清理
    cleanup_candidate_grid_cells_per_axis=2,  # CPU 清理候选网格
    device="cpu",  # 本接口示例；生产 GPU profile 显式 cuda:N
    trace_callback=None,  # 可选只读逐步诊断，不改变更新
    profile=False,  # 不增加生产 CUDA 同步
    target_search_backend="serial",  # 保留目标搜索顺序
)
```

完整七文件、全部后端限制和返回字段见[功能页](../../../../docs/recon_all/PYTHON_PIAL_PLACEMENT.md)。MRI 为 `(X,Y,Z)`、conform voxel 网格；white 为 `(N,3)` surface RAS/mm 和 `(F,3)` 有序面；标签按相同顶点编号，阈值为文本标量。输出 `pial.T1` 保留输入面和几何尾部；报告为原有轨迹/清理/秒数 `dict`，此预检不改变返回结构。

新增内部 `_validate_pial_volume_geometry(brain, wm, aseg, surface_metadata)` 接收已加载的三张 nibabel `MGHImage` 和 white 几何字典，返回原公式 float32 `(4,4)` surface RAS/mm→voxel 变换。只读头；影像像素不复制或重采样。失败抛 `ValueError`，不准备强度、不进入优化、不创建输出目录。既有缺文件、迭代上限和残余相交错误保持。

## 3．命令行与复现

生产优化器没有单独 Python CLI；此预检随原接口执行。只读真实输入检查脚本的全部参数如下，路径均指向同一冻结版本：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 taskset -c 108-111 \
"$FNIT_PYTHON" check_geometry.py \
  --subject-directory "$FNIT_FROZEN_SEVEN_INPUTS" \
  --installed-directory "$FNIT_INSTALLED_TARGET" \
  --baseline-source "$FNIT_BASELINE_PIAL_SOURCE" \
  --candidate-source "$FNIT_CANDIDATE_PIAL_SOURCE" \
  --expected-baseline-sha256 962e36045e99210b5370a4bce1d1204b5f3dc9885e48eeb5e887c7db45518cbe \
  --expected-candidate-sha256 223b538b33ad976545a2561b3fe8ee0584f1695bc8d12d676afa9d049e05e002 \
  --expected-inputs expected_inputs.json \
  --hemisphere lh \
  --output "$FNIT_FRESH_CHECK_JSON"
```

- `--subject-directory`：只读七输入根目录，不能用参考输出选择参数。
- `--installed-directory`：冻结 FNIT 安装，用于导入相同成熟子函数。
- `--baseline-source`、`--candidate-source`：原/新 pial Python 源；两个 `--expected-*-sha256` 在读取真实输入前核对源码。
- `--expected-inputs`：本目录七个相对文件名→SHA-256 JSON，读前/读后均校验。
- `--hemisphere`：必填 `lh/rh`；本次只有左侧当前输入。
- `--output`：尚不存在的安全 JSON；失败不写成功收据。

冷 CLI 用仓库 `tools/time_recon_python_benchmark.py --log-path "$FNIT_NEW_LOG" --report-path "$FNIT_NEW_TIME_JSON" -- ...` 包装以上命令，包含导入、哈希读取、头与网格读取、校验和 JSON 写入。脚本内部 `seconds_*` 截止成功收据写入前，二者不混用。

本地专项：`PYTHONPATH=src python -m pytest -q tests/recon_all/test_pial_volume_geometry.py tests/recon_all/test_pial_rejected_stop.py`。这是小网格 API/控制流测试，不能当作真实 pial benchmark。

## 4．原软件调用与生产依赖

预检属于 `mris_place_surface --pial` 所需输入空间的接口约束，没有对应独立原版 CLI。完整参考命令及固定源码链接见[功能页](../../../../docs/recon_all/PYTHON_PIAL_PLACEMENT.md)。本次没有重新运行官方程序或读取官方坐标，合法输入的变换对比使用原 FNIT 函数。

显式 `backend="python", topology_backend="python-ga"` 的生产后半段走自有四轮 white/pial；defects 用 Torch，曲率用 Torch/Numba，面积/厚度/体积和分区统计用既有自有模块。保留自有 CPU 有序核和通用编译库；未发现该配方漏调用原版阶段程序。原生参考分支仍供独立 benchmark，不能把这一静态审查称完整隔离验收。

## 5．当前真实结果、耗时及验证边界

当前七输入来自公开 T1 的 FNIT 自产冻结末态。原始 white 输入 SHA 为 `df9eaf2ed4e709bee489c93c24e5afbe5c7854f2d7baa04e8103091c3c29f767`；全部输入、实际源码/库版本、矩阵和计时绑定在 [ACTUAL_RECEIPTS.json](ACTUAL_RECEIPTS.json) 与 [SOURCE_BINDING.json](SOURCE_BINDING.json)。

| 本次实际检查 | 结果 |
| --- | --- |
| white 网格 | 95,045 顶点，190,086 面，坐标有限；只读 |
| MRI 网格 | 三张 `(256,256,256)`，scanner affine 完全相同 |
| MRI 自身 tkr | brain/WM 相同；aseg 不同，未用于共享索引 |
| 原/新实际变换 | float32 `(4,4)` 全部字节相同，different/max/P99 = 0 |
| 变换字节 SHA | 双方 `eb9c075f741a95e5310a7b94f837fbab79e9e0f3f5e056c8549ad2d81364f671` |
| 输入稳定性 | 七文件读前/读后 SHA 全相同 |
| 完整冷检查 CLI | 8.837371 秒，含导入、读取、哈希、检查与写出 |
| 最大子进程 CPU RSS | 384,229,376 字节；不是同时父子内存合计 |
| CPU/设备 | 同一已授权计算环境，亲和性108–111，总4线程；CUDA隐藏/未初始化 |
| 本地 API/四轮契约 | 新22项＋既有17项，39通过，pytest 10.54秒 |

没有解码真实 MRI 像素、调用四轮优化器、重测表面误差图或执行新整例。真实检查只有合法输入空间及变换逐位保持证据；完整 pial 精度/速度、整例耗时/显存和整体指标等效未测。此前真实脑图继续按功能页原版本查看，不能改标为此次新输出。

## 6．版本、失败记录与安装

- 原 FNIT pial 源 SHA `962e3604…`；最终候选 `223b538b…`，复用的几何子函数 `0df4ebcc…` 未修改。绑定的 Git base 是 `f3bd14e926520c160b4b585c0370d07771a9afd2`，候选脏源由完整 SHA 识别。
- v1 真实检查拒绝了未消费的跨 MRI tkr 差异；v2 拒绝了有效标记的原格式行尾注释。两次都未进入优化、未写表面；没有改变 MRI、阈值或数值门。实际失败与计时保留在 [FAILURE_HISTORY.json](FAILURE_HISTORY.json)、[ACTUAL_RECEIPTS.json](ACTUAL_RECEIPTS.json)，冻源清单保留在 `frozen/v1`、`frozen/v2`；原样日志/源码保留计算端，不发布私人日志。
- 本地初次基础环境 NumBa 导入失败；另一次扩展测试44通过/1失败，原因是本地缺已声明的 `tifffile`，并非放置算法失败。CFFF 专项因没有 `pytest` 未执行；没有为重复测试改旧环境或下载依赖。最终专属39测试在已有 FNIT Python 环境完成，详见 [LOCAL_CONTRACTS.json](LOCAL_CONTRACTS.json)。
- 无新增运行依赖；nibabel/NumPy/Numba 等沿主页 Conda 声明。未做新的干净 Conda 安装或整例隔离验证。

## 7．参考文献与原实现

- Fischl B. FreeSurfer. *NeuroImage*. 2012;62(2):774–781. DOI：[10.1016/j.neuroimage.2012.01.021](https://doi.org/10.1016/j.neuroimage.2012.01.021)。
- [FreeSurfer 固定源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)，只作为独立参考。
- FNIT `place_surface_geometry.py` 的既有 `surface_ras_to_voxel()`；本次复用其原计算顺序，不引入替代几何公式。
