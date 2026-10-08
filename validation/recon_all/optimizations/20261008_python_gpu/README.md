# 2026-10-08 纯 Python GPU 迁移与真实阶段回归

本目录绑定代码提交 `953680ce0cd31e107e01288543fbdff8fc9e0be3`。本轮没有把旧整例报告改标为当前结果；完整九例基线仍见 [热点收据](../../../../docs/recon_all/HOTSPOT_ACCELERATION_20261007.md)。

## 本轮实际修改

- `finalsurfs_python.run_finalsurfs()` 在显式 CUDA 设备上调用已有 `fix_ento_wm_gpu`，包括独立的 ACJ 语义；CPU 入口保持 `fix_ento_wm`。
- `native_free._run_white_mri_chain()` 将 recon-all 的目标设备传给 `finalsurfs_python`。
- 新增 `backend="python-gpu"` 能力矩阵。N4、GCA/EM、完整 WM/aseg、拓扑、white/pial、缺陷和曲率统计尚无完整 Python/CUDA 等价实现时，入口在创建被试目录前抛出结构化 `PurePythonGpuUnavailable`，不回退到原生程序。
- Jacobian CUDA 内核保留为显式实验，但真实配对显示其完整 114k 顶点映射受启动/I/O 限制，生产仍选择更快的 CPU 路径。

## 真实阶段回归

环境为已授权 `gpucw1` 的 H100 PCIe 80 GB、PyTorch 2.5.1/CUDA 11.8、FNIT Conda 环境、单阶段新进程。输入是公开 ds000114 `sub-07` 的 FNIT 自产 recon-all 中间目录；GPU 和 CPU 分别在独立复制目录中执行，未读取官方参考目录。

| 阶段 | CPU wall (s) | CUDA wall (s) | 差异 | 结果 |
|---|---:|---:|---:|---|
| `brain.finalsurfs`（含加载、mask、EntoWM/ACJ 编辑和写出） | 6.537774 | 6.030862 | CUDA 快 7.75%（单次配对观察） | 256³ uint8 体素差异 0，最大绝对差 0；两份输出与既有 subject 输出 SHA-256 均为 `f3a0be99…390a55` |
| Jacobian 114,342 顶点（显式实验） | 0.078845 | 1.154865 | CUDA 慢 14.64 倍 | 最大绝对差 `9.536743e-7`，P99 `2.384186e-7`；生产不切换 |

GPU 阶段默认保持 TF32 全局策略；该 volume edit 使用整数标签和 FP32 mask，不启用 FP16/BF16。上述是单阶段配对，不是整例提速，也不代表共享节点稳定吞吐。

## 纯 profile 状态

远程依赖环境导入能力矩阵得到 `ready=9`、`blocked=9`、`complete=false`，`native_programs=[]`。因此当前 `--backend python-gpu` 会明确失败并列出阻断阶段；默认 `--backend native` 仍是当前可验收混合流程。不能以当前 2669.520 s 九例中位数宣称达到 600 s；按已有真实基线至少需要约 4.45 倍全流程提速。

## 验证命令与限制

- 本地 `python3 -m py_compile`：通过；本地没有项目 Conda 依赖，未声称本地 pytest 通过。
- 远程 `test_white_mri_device_wiring.py`：2/2 通过。
- 远程能力矩阵断言：通过，确认纯 profile 不解析原生二进制。
- 没有在本提交上重新运行原始 T1 到 138 项整例；因此严格复现、整体指标等效、整例显存和 10 分钟目标均为 `not_assessed`。
