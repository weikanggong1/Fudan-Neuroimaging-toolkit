# 标准 sphere 完整 GPU 收尾接线

## 1. 功能与流程

复用仓库已有 `finish_standard_sphere(device=...)`，仅将完整标准 sphere 收尾从 CPU 切换到显式 CUDA。保留原负面判断、marked/SOAP更新、全顶点径向投影、步长和停止条件，选择已有 dense 算法；实验 marked 加速另列，未设为本入口默认。

```mermaid
flowchart LR
  A[自产 smoothwm] --> B[完整标准 inflated/sulc]
  B --> C[原标准 sphere unfold与线搜索]
  C --> D[既有完整dense收尾 CPU或GPU]
  D --> E[同序sphere及实际轮次报告]
```

## 2. Python 与输入输出

```python
from fnit.recon_all.native_free import run_recon_all_python

reconstruction_report = run_recon_all_python(
    t1="/data/public_T1w.nii.gz",  # 一幅原始真实T1，NIfTI或MGH
    subject_dir="/data/recon/new_subject",  # 空输出目录
    weights_dir="/data/fnit/weights",  # 已校验的声明权重
    assets_dir="/data/fnit/assets",  # 已校验的模板/图谱
    native_bin_dir="/data/fnit/native/bin",  # 独立源码编译程序
    device="cuda:0",  # 显式目标GPU，不回退CPU
    threads=4,  # 总预算，两个表面worker各2线程
    hemisphere_workers=2,  # 私有目录的双侧exec，父CUDA和精度保持
    inflate_backend="torch",  # 完整GPU标准inflation，nofix保持原流程
    sphere_finish_backend="torch",  # 完整dense GPU收尾；默认cpu
    profile_stages=True,  # 剖析同步指定GPU，墙钟包含加载/搬运/写出
)
```

新增 `sphere_finish_backend` 为 `cpu`（默认）或 `torch`。torch要求 `cuda:N` 和两个隔离半球worker，仅 surface exec 局部启用分配缓存；其他组及父进程策略保持。完整参数、原始T1/conform/scanner RAS及surface RAS/mm约定沿用[主接口](../../../../docs/recon_all/README.md)。其余 sphere 目标函数、标准法向和 sphere.reg 算法不随该选项改变。返回原输出清单、球面逐阶段报告与实际 `finish_backend/finish_device`；非法组合在输出和线程设置前报错，计算失败传播，不静默回退。

内部 `_run_accurate_sphere_pair` 的 `subject/hemi` 指自产单半球同序 smoothwm，`assets/inflate_binary` 为声明资源，`device` 为主设备，`normals_backend` 默认numba，`inflate_backend` 默认native；新增 `sphere_finish_backend` 默认cpu。返回inflation/sphere读写秒数和完整sphere报告；生成inflated/sulc/sphere，不改变RAS/mm或有序面。私有目录的发布仍由原worker处理。

## 3. CLI 与复现

`fnit-recon-all --sphere-finish-backend torch --device cuda:0 --hemisphere-workers 2` 原样选择该后端；其余必需原始T1/输出/权重/资产参数见主入口。批量API和 `tools/benchmark_recon_torch_end_to_end.py` 转发并记录同一选择。

```bash
python validation/recon_all/optimizations/20261009_sphere_finish_integration/run_contracts.py \
  --source-root /path/to/frozen/source \
  --code-version BASE_COMMIT_PLUS_SOURCE_HASHES \
  --output-report /path/to/new/contracts.json
```

`source-root` 是冻结 src/tests/tools；`code-version` 是实际基线提交与覆盖SHA；`output-report` 必须是新JSON。输入/导入失败抛异常，测试失败返回非零并保留逐项日志。本脚本只测入口契约，不以模拟数据代替真实MRI benchmark。

### e34 提交的实际安装和入口回归

冻结 `e34a1829145b34158f7b37b5d3046f8b02c03617` 已在现有声明的 Conda 环境完成独立 wheel 构建（14.761秒）、私有 target 安装（2.479秒）和实际已安装 CLI 检查（1.992秒）。13份相关模块安装前后字节相同，包括完整sphere收尾与marked候选；已安装CLI包含 `--sphere-finish-backend`。wheel为5,015,256字节、SHA-256 `b146a3cd80383d6b601122352ab57c7f96b98c96ad7a58500b7da2e26d925684`。同提交45项入口契约通过。见[完整安装收据](package_e34/report.json)、[源码身份](package_e34/VALIDATION_SOURCE_PROVENANCE.json)及[契约报告](package_e34/contracts.json)。这是现有环境的无依赖安装/接口验证，未执行安装产物MRI推理，不当作全新Conda或物理隔离整例。

## 4. 原软件对应

复用 FNIT 已有 standard sphere；对应 `mris_sphere` 的内部末尾 overlap cleanup，没有独立官方收尾CLI。完整命令、固定源码和参数见[标准sphere及本轮收尾说明](../../../../docs/recon_all/SPHERE_FINISH_TORCH_20261009.md)。不读取官方结果，不省略投影或减少1001轮上限。

## 5. 本次验证

45项入口契约通过，实测2.343秒；新增6项覆盖前置设备/worker拒绝、实际leaf只选择finish_device、exec/API/CLI/batch/benchmark转发。报告不是MRI精度或速度证据。

真实sub07左右半球同输入完整收尾及实际根函数双侧冷进程链已另外验证，见[完整阶段与生产接线链](../../../../docs/recon_all/SPHERE_FINISH_TORCH_20261009.md)：链245.389→173.390秒，缩短29.34%，包含copy/exec/import/JIT/IO/publish及透明观察开销。unfold168/189轮、收尾1001/0轮的坐标、标记、SOAP、线搜索及停止状态一致，inflated/sulc/sphere文件几何和footer一致。左侧既有47个负面、右侧0，没有新增；这不代表所有几何质量项目通过。该阶段收益不能算原始T1整例提速，整体指标等效仍 `not_assessed`。

## 6. 版本绑定

45项测试基于 `765c0fe95fc2897273a999a2d09ba03ce55a4c9f` 加本次根入口覆盖，见 [sphere_source.json](reports/sphere_source.json) 与 [sphere_contracts.json](reports/sphere_contracts.json)。冻源码的成熟 dense finish SHA为 `9ba0ce2e7dd6ab0a731a45266f3e7fd8e67b48a39ed7117c65d68cc2a74f9825`；未使用同时开发的marked候选。实际双侧链绑定同一根入口SHA，完整收据另列，不能把以后提交追标到本次运行。

## 7. 参考与原实现

- [FNIT完整收尾算法、真实链与上游参考](../../../../docs/recon_all/SPHERE_FINISH_TORCH_20261009.md)。
- [recon-all原接口和参考文献](../../../../docs/recon_all/README.md)。
