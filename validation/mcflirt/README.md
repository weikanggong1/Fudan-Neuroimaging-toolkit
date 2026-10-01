# MCFLIRT 真实帧控制

`first8_cpu.public.json`、`first8_cuda.public.json` 及 `first8_warm.public.json` 记录同一例真实 UKB BOLD/SBRef 的前 8 帧估计，参照为同数据的原 MCFLIRT 完整 490 帧运行。脑内距离只比较前 7 帧：原代码 8 mm 阶段没有给最后一帧传递初值，截取片段会改变最后一帧的实验条件。

报告只有匿名标量、源文件 SHA-256 和输入 SHA-256。每帧矩阵保存在服务器的私有文件中，未上传原始影像或矩阵数组。JSON 中 source hash 指向当时实际测量的文件；随后增加的输出接口及检查没有用于该次测量。`first8_warm.public.json` 的同进程第二次估计观察为 CPU 5.41 秒、CUDA 11.13 秒；共享服务器负载未控制，两个 device 内各自重复的矩阵完全相同。完整链的最终源码和结果以[全流程记录](../fmri/matched_native.md)为准。

复测时先在服务器完成原 FSL MCFLIRT 运行，再执行 FNIT 的独立控制：

```bash
# 原数据仍在服务器；8帧control只输出匿名报告和私有矩阵，不生成图像。
python validation/mcflirt/compare_motion.py \
  --bold "$RAW_BOLD" --reference "$SBREF" --brain-mask "$EPI_MASK" \
  --original-matrices "$ORIGINAL_MCFLIRT_MATRICES" \
  --output-dir "$FRESH_CONTROL_OUTPUT" --device cuda:1 --frames 8
```

不要把小样本冷运行当成完整运动校正的速度。CUDA 第一次执行会编译固定尺寸 NCC 归约；原 MCFLIRT 的命令耗时还包含最终重采样及写盘，不能直接与只估计矩阵的计时相除。

`native_cost_probe.cc` 是仅用于验证的独立小程序：链接安装好的 NEWIMAGE，读取真实帧，按 MCFLIRT 的 8/4 mm 参考重采样计算四个固定平移矩阵的 NCC。它没有复制原软件代码，也不被 FNIT 运行时调用。构建需要原 FSL 开发头文件与库，使用原软件做 benchmark 时才需要。对应输入和重采样输出均应留在私有验证目录。
