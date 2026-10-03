# 用户模板与矩形 SC：组件验证

## 实际范围

本记录绑定首个实现 `ea3cb054` 的三个计算模块，逐文件 SHA 见 [real_input_preflight.public.json](real_input_preflight.public.json)。完成：

- gpucw1 上 CPU、4 线程，原矩阵/表面/节点与新 paired-template 回归 **30 passed，2.89 s**；CUDA 对该进程不可见，未初始化 GPU。[测试记录](unit_tests.public.json)。
- 十例公开 ds001226 的真实 recon-all 输入预检：CON01、CON03–CON11；每例读取双半球 aparc/a2009s `.annot` 及 `aparc+aseg.mgz` / `aparc.a2009s+aseg.mgz`。检查整数、有限值、表面顶点数、完整 LUT 和原始文件 SHA。
- CON01 四张模板实际准备到既有 corrected-DWI 的 `96×96×60` 网格：使用该次运行已保存的 DWI→T1 scanner-RAS 变换，未重新配准。

没有运行 TOPUP/EDDY、recon-all、tracking 或 SIFT2，也未以该组件预检替代十人端到端 benchmark。没有本次新 pair 对官方的独立数值误差或 GPU 加速比。

## 真实模板准备结果

| CON01 输入 | 完整节点数 | DWI 中出现节点数 | 非背景 DWI 体素数 | CPU 实际墙钟 |
|---|---:|---:|---:|---:|
| native aparc，双半球 surface | 70 | 68 | 30,802 | 41.983 s |
| native a2009s，双半球 surface | 150 | 148 | 30,802 | 41.920 s |
| 原 `aparc+aseg.mgz` 全非零标签 | 104 | 104 | 101,431 | 2.200 s |
| 原 `aparc.a2009s+aseg.mgz` 全非零标签 | 184 | 184 | 101,431 | 2.256 s |

所有输出为有限的 int32 连续 `0..K` 标签，affine 逐值保持实际 corrected-DWI affine。Surface 的两个未出现声明节点仍占据矩阵行/列。此处 volume 示例明确包含分割中全部非背景结构标签；它与标准筛选的 84 节点 fs-aparc atlas 的定义不同，不能把不同节点数的矩阵直接用于精度比较。

以上时间包括读取、标签编号、surface projection 和 DWI 最近邻准备，设备为 CPU；仅有一次测量，未与官方程序进行配对计时。它不代表 GPU 模板准备或整个 SC 的运行时间。

## 实际脑图

![CON01 真实 native T1 与两张原始体积模板](real_template_inputs.png)

图示为实际 CON01 recon-all T1 与体积模板中的皮层标签，在原生同一网格显示。这里展示输入空间；DWI 端点分配及 paired SC 结果由全管线评测记录。图由 [plot_real_template_inputs.py](plot_real_template_inputs.py) 读取真实文件生成，使用项目现有 Pillow。

## 可重运行的命令

工作目录应为含本次源码 `src/` 的独立工作树，`input_bindings.json` 是协调者验证过、记录十例实际文件位置的私有绑定表；不将私有路径发布到仓库。

```bash
# 相同 CPU 测试；CUDA 不可见确保不与 GPU 计时任务竞争。
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH=src python -m pytest -q \
  tests/connectome/test_template_pairs.py \
  tests/connectome/test_assignment.py \
  tests/connectome/test_atlas_surface.py \
  tests/connectome/test_freesurfer_subject.py

# 实际十人预检和 CON01 四模板准备。
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTHONPATH=src python validation/connectome/template_pairs_20261003/task_02/read_real_inputs.py \
  --bindings input_bindings.json --output real_input_preflight.json

# 实际输入脑图。
CUDA_VISIBLE_DEVICES="" python \
  validation/connectome/template_pairs_20261003/task_02/plot_real_template_inputs.py \
  --bindings input_bindings.json --output real_template_inputs.png
```

矩阵科学定义、所有模板空间、两方向重叠规则、原软件对应步骤与 API 见[用户说明](../../../../docs/connectome/template_pairs.md)。
