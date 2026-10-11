# 独立 Python 候选的 wheel 安装产物检查

## 1. 功能和实测范围

2026-10-10从实际工作树冻结1417个源码与安装文件，构建wheel并安装到新的独立target目录。源绑定为`435a1d6ba64588a286be2cdf94cae8de4a2dc23a`加当时未提交源码；各文件SHA见[SOURCE_BINDING.json](SOURCE_BINDING.json)。源码快照中的Python配方有14项固定选项，未包含随后扩充的21项配方，不能把本报告改标为后者的安装验证。

本次检查固定SimpleITK依赖、CLI入口和8个模块的独立安装路径与源码SHA。既有Conda环境提供Torch等依赖，SimpleITK/tifffile由临时依赖目录提供。没有创建干净Conda环境、运行MRI、执行GPU推理或追踪完整生产进程；不判断整例精度、耗时或独立部署通过。

## 2. Python调用、输入和输出

```python
from pathlib import Path
from run_wheel_checks import run_wheel_checks

installation_report = run_wheel_checks(
    repo=Path("/work/FNIT"),  # 实际冻结源码工作树，记录Git及未提交文件SHA
    output=Path("/tmp/fnit-wheel-proof"),  # 尚不存在的独立目录，不改共享环境
    dependency_path=Path("/work/python-extra-libs"),  # 已有SimpleITK等库；可为None
)
```

`repo`和`output`必填，`dependency_path`默认None；均无空间或物理单位。解释器须已有项目声明依赖及pip/setuptools/wheel，脚本不下载依赖。输出含源码快照、wheel、新target、日志及`SOURCE_BINDING/BUILD/INSTALL/CHECK.private.json`；返回安装检查字典。私有报告含本机路径，发布前应脱敏。已有输出目录、缺依赖、构建/导入失败会报错并保留日志。

## 3. 命令行

```bash
python run_wheel_checks.py \
  --repo /work/FNIT \
  --output /tmp/fnit-wheel-proof \
  --dependency-path /work/python-extra-libs
```

三个具名参数与Python接口相同。实际构建使用`pip wheel --no-deps --no-build-isolation`，独立安装使用`pip install --no-deps --target`，检查在新的Python进程内执行。源码与现有环境保持原位置；仅写新任务目录。

## 4. 原软件调用

wheel安装检查没有对应FreeSurfer阶段命令。本次没有生成官方MRI参考。包构建会编译FNIT的MSM扩展，该扩展用于另一功能，本次recon-all导入检查没有执行它；不能将包内自有扩展视为调用原版FreeSurfer程序。

## 5. 本次结果和时间

实际产物为`fudan_neuroimaging_toolkit-0.16.0-cp311-cp311-linux_x86_64.whl`，5,114,994字节，SHA-256：`faff92bb6cf0a1920d8e6eb5fcefda33569a398afdc1b5124d1d6e79158d8989`。未把wheel二进制上传到此目录。

| 检查 | 结果 | 实测范围 |
| --- | --- | --- |
| 源码冻结和哈希 | 1417文件、前后Git HEAD相同 | 32.442秒，含复制及哈希 |
| pip wheel | exit0 | 6.284秒，含MSM自有扩展构建 |
| 独立target安装 | exit0 | 0.792秒，无依赖解析/下载 |
| 元数据、8模块与CLI | 全部通过 | 3.923秒，含新进程CLI帮助 |
| 固定依赖 | `SimpleITK==2.5.6` | 真实导入版本2.5.6 |
| CLI入口 | `fnit.recon_all.native_free:main` | `fnit-recon-all --help`退出0 |
| CUDA初始化 | False | 只导入，不是零显存测量 |

8个模块覆盖Python配方、N4、输入链、主入口、white.preaparc、最终white、pial和数字拓扑；安装模块SHA与快照逐一相同。详见[CHECK.json](CHECK.json)、[构建](BUILD.json)和[安装](INSTALL.json)。这些时钟是安装阶段，不能相加当作recon-all整例耗时。MRI精度、资源峰值及脑图本次未测。

## 6. 版本和报告绑定

本次实际Git与源码绑定、wheelSHA、旧14项配方和依赖保留原值；随后21项配方应使用新的构建收据。公开文件只替换本机绝对路径，数值、bool、null及其类型不变。[PUBLIC_EXPORT_MANIFEST.json](PUBLIC_EXPORT_MANIFEST.json)分别记录私有原始SHA和公开SHA，不能混用两者。

## 7. 安装声明与相关实现

- [项目依赖和入口](../../../../pyproject.toml)、[构建脚本](../../../../setup.py)。
- [独立SimpleITK N4](../../../../docs/recon_all/N4_SIMPLEITK.md)。
- [完整自有Python流程和整例状态](../../../../docs/recon_all/PYTHON_SURFACE_PIPELINE.md)。
- [复现脚本](run_wheel_checks.py)。此脚本复用FNIT现有模块，不引入原版阶段程序。
