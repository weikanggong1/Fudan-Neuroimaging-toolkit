# 2026-10-08 PyTorch 原生阶段迁移收据

本收据绑定当前分支的真实 sub-07 阶段测试。GCA 的 PyTorch 候选评分在 H100 上与 CPU 搜索产生相同 LTA，并把同一 LTA 送入归一化后得到逐体素相同的 `norm.mgz`；EM 精修仍是 FNIT Python FP32，因此这不是完整 GPU EM。

WM segmentation 的 Torch/CPU 两种输出逐体素相同，但相对已有参考仍有 64 个标签体素差异（Dice 0.9999401770），所以该后端默认保持显式 opt-in。GPU 重跑受到共享节点 OOM 影响，没有发布 WM 的速度结论。

表面几何模块只作为独立 PyTorch 内核和回归测试提供，尚未接入 topology/remesh/white/pial 生产链。N4 的不等价草稿已删除，生产仍使用 ITK C++。
