# 准备后完整五项评分：本次真实回归

对应 [中文接口说明](../../../../docs/recon_all/TOPOLOGY_PATCH_FITNESS_ASSEMBLY.md)。只验证同一真实候选的完整五项评分：不称上游 MRI/主曲率重生、完整 GA 或原始 T1 整例通过。

## 当前结果

- [v4 完整 API](reports/v4/report.json)：execution 完成、评分输出及自身重复逐位相同；五个核心 TP、double fitness、完整自产距离、全顶点反馈、8个 unMRI 返回字段、4个 NaN 原 bits 都相同。
- 准备矩阵16个数值相同，7个元素正负零的位模式不同；**overall_internal_byte_reproduction=False**。矩阵字节诊断原样保留，未修改成熟变换或数值容差。
- [外部启动器](reports/v4/driver_receipt.json)：实际 child exit0，完整冷子进程 10.61987467110157 s，含加载/两次 API/比较/全部写出。
- [v3 原失败](reports/v3/report.json)：相同评分全完成，但新工具合并输入矩阵字节门导致 exit1；原数值、SHA与墙钟未覆盖。
- [v2 启动失败](reports/v2/launcher_failure.json)：launcher newline 转义 SyntaxError，未开始评分，不归算法、CPU/GPU或显存。

完整 API 首次 6.176419661380351 s，重复 0.15620194375514984 s。首次包含当次 JIT；共享环境不能按两次之比宣称优化提速。参考观察器完整进程约6秒，包含不同的初始化与捕获，未测可比的独立五项 API。

## 固定输入与版本

公开 ds000114 sub-07/ses-test、FNIT 自产79a41cdd LH，first complete candidate：110534顶点、216860面、patch99顶点/138面、局部体积21×19×16。原始MRI和capture保留私有诊断路径，公开目录只含去敏报告和派生脑图。

- [24文件候选源码 manifest](reports/v4/source_manifest.json)：观察 HEAD `436a3db7dbfe5032ed26f8875ae1f3640954b497` + 指定未提交新增接口/工具 SHA。
- 评分接口 SHA `59111c5f764d535625276402b756c6f9240a87ab5396886e81606f93df074827`；完整工具 SHA `fc3666fd1aa4d6ff9b8a0571f14f7c5ff5d390f3ba39b2bb6cd47d625bd33478`。
- 复用完整距离 `66b84289...` 和正式 angle `562d7f5f617048b19d133fc74ebd4ed1800a4932f168a010a1c74ce54d76ddd0`，全部实际SHA见 manifest。
- [参考0](reports/reference_repeat0.json)、[参考1](reports/reference_repeat1.json)：源 d932c45，program `2df26fc8f997e13eee8d1dd97c46b452e2d25980645927b1d4de31f2eec9bec7`，同一完整 first fit，两次 capture/字段稳定；不使用初始全图统计代替候选。
- [导出 manifest](EXPORT_MANIFEST.json)：仅将私有目录前缀和主机名归一化，全部数值、位模式和原始私有SHA保留。
- [payload manifest](payload_manifest.json) 绑定当前公共源码、工具、测试、说明和报告字节。

## 重现

详见中文说明第3节的完整具名 CLI。`statistics/volume/distance` 是明确的冻结准备后阶段输入；`fnit_joint_pdf` 来自 FNIT 初始校准。每次使用空的新输出目录，CPU4线程，先落盘候选再比较参考。

```bash
PYTHONPATH=src python -m pytest -q \
  tests/recon_all/test_topology_patch_fitness.py \
  tests/recon_all/test_topology_patch_fitness_benchmark.py
```

图脚本 [plot_prepared_fitness.py](scripts/plot_prepared_fitness.py) 读取私有同输入检查点和已完成候选，写公开衍生图与哈希；[图收据](reports/figure_receipt.json) 绑定2.412038325332105s加载/绘制/写出的实际版本。它不生成生产输入或评估最终网格质量。

## 保留的范围限制

候选 SFF、局部MRI/log-P目前在本次组装中是准备字段，不能说是该接口自产；后续各自独立转写/回归才可接入其上游。校准直方图也沿控制器初始阶段传入。本文无完整GA时间、生产默认改变、显存达标或整体指标等效结论。通用 Python/NumPy/Numba 运行库不是原版 FreeSurfer 阶段程序。
