# 独立 Python recon-all 安装与运行预检

## 1. 功能

主页 Conda 环境提供完整 Python 配方的 PyTorch、Numba、SimpleITK、TIFF 读写与可选 NVML。`check_python_runtime` 只核对依赖是否能加载和固定 core 资产是否完整，不发现原版程序，不初始化 CUDA，不启动重建。完整整例隔离验证单独报告。当前 Python profile 要求显式 `cuda:N`，线程预算至少2；CPU通用核仍可执行。完整 recon-all 的 API、批处理与测评入口均在校验阶段拒绝 CPU 设备；MNI只使用已有GPU后处理，不发现、读取或执行原版warp程序。独立MNI函数同样不支持CPU或conda后端，错误发生在影像读写前。

## 2. Python 调用、输入和输出

```python
from fnit.recon_all.python_runtime import check_python_runtime

runtime_report = check_python_runtime(
    require_nvml=True,  # True：显存监测必须可导入；False：缺失只作为可选项记录
    assets_dir="/data/fnit_assets",  # 固定core模板目录；None只检查依赖
)
if runtime_report["status"] != "passed":
    raise RuntimeError(runtime_report["missing"])
```

返回字典含 status、各依赖的版本/可加载状态/模块哈希、missing、incompatible、资产校验状态及 `whole_isolation=not_assessed`。版本号无单位。资产以原 manifest 的文件名、字节数和 SHA-256 标识；不输出服务器路径或账号。失败不会回退到原程序。

## 3. 安装和命令行

```bash
conda env create -f environment.yml
conda activate fnit
# 在其他已满足主页CUDA/PyTorch版本的环境，补装声明的Python配方附加依赖：
python -m pip install -e '.[recon-all-python-stages]'
fnit-verify-recon-all-python --assets-dir /data/fnit_assets --require-nvml \
  --output /data/results/runtime.json
fnit-setup-recon-all-assets --dest /data/fnit_assets --verify-only
fnit-setup-weights --dest /data/fnit_weights --model recon-all --verify-only
```

`--output` 为 typed JSON 文件；默认只写 stdout。`--assets-dir` 可省略，省略时不检查资产。`--require-nvml` 默认关闭；只检查 Python 依赖，不证明 NVML 的 PID 与当前 namespace 可正确关联。预检失败时 CLI 返回 1，成功返回 0。Python 生产配方无需 native_bin_dir、系统 FreeSurfer 或 FS_LICENSE；参考程序仅由独立 benchmark 使用。SimpleITK 为固定 2.5.6；PyTorch/Triton 版本沿用主页 Conda 配对。TIFF 读取需要 tifffile 和 imagecodecs，NVML 包名为 nvidia-ml-py。

资产使用当前固定 Release 清单及逐文件许可路线；不下载或镜像缺少明确再分发许可的资源。权重预检复用现有工具。core 单 T1 模板由 `fnit-setup-recon-all-assets` 现有默认选择提供，`--all` 属于更大的参考资产范围。

## 4. 原软件调用

预检没有独立 FreeSurfer 对应命令。完整流程的参考为 `recon-all -all`；MNI 参考转换/求逆/检查图和表面统计对应命令见各功能页，只用于独立对照。

## 5. 真实验证

本轮同输入、同表面版本统计与完整 MNI 验证由 [任务记录](../../validation/recon_all/optimizations/20261011_metrics_install/README.md)绑定源码、输入、资产与参考程序 SHA。两组真实完整阶段已通过，9边界测试、离线安装/入口通过；参考ELF迁移差异在报告保留。不得将导入、PATH 或 ldd 当作整例隔离证明。

总控另对集成提交 `e9e89cac07cc65ef37f8c3223e53f03968f2af05` 的Git冻结源码完成本地离线wheel回归：构建2.86秒、独立目录安装1.96秒、完整入口检查9.99秒，其中fresh依赖console为2.24秒；16模块的源码、wheel和安装文件SHA一致。依赖API及实际console均通过，调用前后TF32设置不变，CUDA未初始化。第一次检查发现本地验证环境缺少主页已声明的tifffile，失败报告保留；补齐到独立依赖目录后复测通过，没有修改共享Conda环境。这不是当前整例、干净Conda安装或动态库/文件访问隔离验收。源码、检查工具及两次实际wheel哈希分别见[集成安装记录](../../validation/recon_all/optimizations/20261011_integrated_python_install/README.md)。

## 6. 版本与 benchmark

2026-10-11：补 TIFF/NVML 的 Python 附加依赖声明和无原版程序发现的运行预检；MNI 库默认仅使用已有 GPU 后处理，原版路径参数默认为 None，原参考实现留在独立验证目录。表面缓存拒绝未授权半精度/整数转换，主曲率仍按已有 TF32 策略计算并恢复调用者设置。

同日集成安装复核：复用已有wheel验证工具，移除过时的21键检查，改为具名配方校验；增加完整依赖与实际安装console验证，科学阶段计算不变。

## 7. 参考

[SimpleITK](https://simpleitk.org/)、[nibabel](https://nipy.org/nibabel/)、[TIFFfile](https://github.com/cgohlke/tifffile)、[NVML Python binding](https://pypi.org/project/nvidia-ml-py/)、[FreeSurfer 固定源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
