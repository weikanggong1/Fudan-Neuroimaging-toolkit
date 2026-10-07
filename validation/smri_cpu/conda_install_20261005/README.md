# CPU 修复后的 Conda 安装检查

## 本次修改

SynthMorph CPU joint 使用独立 Eigen 3.4.0 头文件。完整主页环境原先将 PyVista/VTK 也交给 Conda：新 VTK 要求 Eigen 5，而可用旧 VTK 的 Qt 依赖又与 Connectome Workbench 冲突，完整求解失败。

保留 Eigen 3.4.0 和原 Torch/CUDA/编译器约束，将绘图包固定为官方 PyPI wheel：PyVista 0.49.0、VTK 9.5.2。这项修改只影响新环境的安装来源；没有修改运行中的环境或计算函数。

## 一键安装入口

```bash
# 在 FNIT 仓库根目录执行，Conda 先安装计算依赖，再安装声明的 pip wheel 和 FNIT。
conda env create -f environment.yml
conda activate fnit
```

权重、模板仍按主页资源配置入口单独下载与校验。本次没有增加原软件安装或复制其程序。

## 已验证与未验证

[solve.public.json](solve.public.json) 保存配置 SHA、实际依赖计划和 wheel URL/SHA。完整 Conda dry-run 返回 277 个依赖，包含 Eigen 3.4.0、GCC/GXX 11.4、Torch 2.5.1、CUDA 11.8 和 Workbench 2.1.0；退出码 0。官方 Workbench 和 Eigen 对应包 URL 均已返回 HTTP 200。独立 pip dry-run 可解析两个固定绘图 wheel，复用已有数值环境。

dry-run 不执行安装，也不验证 pip 与全新 Conda 环境的全部联合导入。本次没有声明干净环境的整仓库功能或 MRI benchmark 通过。自有平均器的 wheel 安装、实际 C++ 编译与逐位合同检查另见[打包记录](../average_cpu_persistent_20261004/packaging.public.json)。

## 来源

[PyVista 官方安装文档](https://docs.pyvista.org/getting-started/installation.html)、[PyVista 0.49.0](https://pypi.org/project/pyvista/0.49.0/)、[VTK 9.5.2](https://pypi.org/project/vtk/9.5.2/)。本次使用上游 wheel，不将其二进制提交到 FNIT。
