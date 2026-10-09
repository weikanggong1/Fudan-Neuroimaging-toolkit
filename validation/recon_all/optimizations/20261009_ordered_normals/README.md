# 有序 Torch 法向阶段验证（2026-10-09）

当前交付是内部法向阶段及显式 sphere 候选入口。完整 GA、remesh/inflate、
相交修复、white/pial 未被此内核替代；默认生产后端不变。

## 收据

- `cpu_real_mesh_abba_final.json`：两例真实 T1 的12张网格，4线程、3轮ABBA/BAAB，
  最大分量误差 `1.7881393432617188e-7`，预设容差 `5e-7`，12/12通过。
- `cpu_unit_tests.txt`：最终代码的CPU默认/缓存/CSR/Torch回归18项通过，CUDA1项
  单列为deselected，完整CUDA尝试失败见初始化收据，不计作通过。
- `cuda_initialization_failure.json`：两次单元回归及最小CUDART探针失败；
  根因未确定，不报告GPU精度或速度。
- `benchmark.py`：冻结同输入的可复现脚本，包含I/O、拓扑准备、冷JIT、
  热CPU、resident Torch以及含搬运API的独立计时。

报告绑定实际基线commit及模块SHA-256；源码修改尚未替换默认流程。
实际官方内部算子、原始T1整例、脑区统计、进程显存采样及隔离安装未执行。

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
