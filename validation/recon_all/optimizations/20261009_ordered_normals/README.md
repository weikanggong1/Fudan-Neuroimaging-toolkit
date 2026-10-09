# 有序 Torch 法向阶段验证（2026-10-09）

当前交付是内部法向阶段及显式 sphere 候选入口。完整 GA、remesh/inflate、
相交修复、white/pial 未被此内核替代；默认生产后端不变。

## 收据

- `cpu_real_mesh_abba_final.json`：两例真实 T1 的12张网格，4线程、3轮ABBA/BAAB，
  最大分量误差 `1.7881393432617188e-7`，预设容差 `5e-7`，12/12通过。
- `cpu_unit_tests.txt`：最终代码的CPU默认/缓存/CSR/Torch回归18项通过，CUDA1项
  单列为deselected，完整CUDA尝试失败见初始化收据，不计作通过。
- `cuda_initialization_failure.json`：两次单元回归及最小CUDART探针失败；
  根因未确定；后续同环境复测成功不用于解释该失败。
- `gpu_real_mesh_abba_retry.json`：同一default环境、GPU0独立窗口的两例双侧
  orig复测，4/4法向与有序Numba逐位相同。4线程、3轮ABBA/BAAB，
  GPU驻留计算中位数4.66–5.69ms，含搬运API5.30–6.52ms；Numba API
  132.75–157.61ms。PyTorch峰值allocated42,039,296B、reserved50,331,648B。
- `gpu_raw_bits_probe.json`：GPU1检查上述4网格原始FP32位模式，4/4完全相同，
  signed zero未被数值相等掩盖；该短探针不替代3轮配对计时。
- `full_sphere_sub07_lh/{baseline,candidate,comparison,quality}.json`：同输入完整
  sphere，最终文件SHA、faces、coords和182次接受步长相同；含I/O观察为
  170.63→133.57s。共享负载及首次JIT影响仍需更多配对。既有49个残余负面
  的面积0.0055923mm²，新旧均相同，网格质量不记作全通过。
- `benchmark_sphere.py`：冻结整步配对，全部加载算法模块SHA、目标GPU同步、
  4线程、含I/O墙钟。`inspect_sphere_quality.py`另外量化负面与finish未收敛。
- `benchmark.py`：冻结同输入的可复现脚本，包含I/O、拓扑准备、冷JIT、
  热CPU、resident Torch以及含搬运API的独立计时。

报告绑定实际基线commit及模块SHA-256；源码修改尚未替换默认流程。
实际官方内部算子、原始T1整例、脑区统计、进程显存采样及隔离安装未执行。
对应自产输入的官方完整sphere未取得，完整RH及第二例sphere仍待验证。

## 复现

```bash
PYTHONPATH=src python validation/recon_all/optimizations/20261009_ordered_normals/benchmark.py \
  --input /data/sub06/surf/lh.orig /data/sub06/surf/rh.orig \
          /data/sub07/surf/lh.orig /data/sub07/surf/rh.orig \
  --output /data/new-diagnostic/ordered-normal-gpu.json \
  --device cuda:0 \
  --threads 4 \
  --repeats 3 \
  --code-version ACTUAL_COMMIT_AND_DIRTY_PATCH_HASH
```

`--input`为冻结有序三角表面；`--output`为新报告；`--device`显式选择CUDA，
也支持CPU诊断；`--threads`固定两后端线程数；`--repeats`是ABBA/BAAB轮数；
`--code-version`填写实际commit及工作树补丁版本，模块哈希由脚本直接读取。
GPU计时会同步所指定设备；显存仅记录PyTorch峰值，不能代替父子进程同时占用。

完整中文接口及范围见 [Torch有序法向](../../../../docs/recon_all/SURFACE_NORMALS_TORCH_20261009.md)。

## 完整sphere复现

```bash
PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
python validation/recon_all/optimizations/20261009_ordered_normals/benchmark_sphere.py \
  --mode baseline \
  --inflated /data/sub07/surf/lh.inflated \
  --smoothwm /data/sub07/surf/lh.smoothwm \
  --output /data/new-diagnostic/sphere \
  --threads 4 \
  --code-version ACTUAL_COMMIT_AND_DIRTY_PATCH_HASH
```

基线结束后相同命令改`--mode candidate --normals-device cuda:1`，其余参数保持
不变；候选结束后改`--mode compare`写比较JSON。`--mode`必填；
`--inflated/--smoothwm`是同序同面的冻结网格；`--output`是独立新目录；
`--threads`默认4；`--code-version`记录实际代码；`--normals-device`默认cuda:0，
只用于候选；`--official-sphere`可选，仅在相同输入出处已确认时解释为官方
同输入对照。候选分配CUDA失败直接报错，不跳过或回退CPU。
