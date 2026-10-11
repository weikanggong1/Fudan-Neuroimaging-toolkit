# 集成Python配方的离线安装与入口回归

## 1. 功能与范围

复用 `tools/verify_recon_genetic_profile_install.py` 核对实际冻结源码、构建wheel、安装到新目录，并执行真实导入、CLI help及安装后的依赖console。这里没有影像计算，不报告被试耗时、GPU显存或整例隔离通过。

## 2. 输入、输出与Python调用

输入为Git `e9e89cac07cc65ef37f8c3223e53f03968f2af05` 的包源码和逐文件清单，没有工作树overlay。清单1470项均为相对路径，记录字节数与SHA-256；没有MRI、权重或许可证内容。源码读取后复制到本次构建目录，原冻结目录不改写。

```python
from pathlib import Path
from verify_recon_genetic_profile_install import verify  # tools目录加入PYTHONPATH

installation_report = verify(
    source_directory=Path("/data/fnit-frozen-source"),  # 冻结包源码，含src/与安装元数据
    source_manifest=Path("/data/SOURCE_MANIFEST.json"),  # files逐文件记录bytes/sha256
    output_directory=Path("/data/new-install-regression"),  # 尚不存在，且不位于冻结源码内部
    python_executable=Path("/conda/envs/fnit/bin/python"),  # 已装主页依赖、setuptools和wheel
)
```

四项均为必填 `Path`，没有默认目录或坐标空间。输出 `report.json` 记录构建、安装、入口墙钟，16模块源码/wheel/安装哈希，依赖和精度状态；目录另保存构建副本、wheel、安装目标、入口JSON及私有日志。非法清单、已有输出、缺依赖或子进程失败抛异常，原异常和日志不替换为成功。NVML仅为可选依赖，不查询驱动；资产、MRI和CUDA计算均不运行。

## 3. 命令行

```bash
python tools/verify_recon_genetic_profile_install.py \
  --source-directory /data/fnit-frozen-source \
  --source-manifest /data/SOURCE_MANIFEST.json \
  --output-directory /data/new-install-regression \
  --python-executable /conda/envs/fnit/bin/python
```

参数与Python接口同义。构建和安装使用 `--no-deps --no-index`，依赖须先按主页环境安装；源码或网络下载不发生。实际安装console的Python路径以新安装目录为首，避免误读editable源码。

## 4. 原软件对应

安装入口审计没有FreeSurfer等价命令；它不运行参考软件。真实阶段参考命令、影像输入和数值对照见各功能页及本轮MNI/统计报告。

## 5. 本次实测

| 范围 | 秒 | 结果 |
|---|---:|---|
| 离线构建wheel | 2.8586 | EXIT0 |
| 独立目录安装 | 1.9594 | EXIT0 |
| 导入、配方、N4/完整依赖、三项fresh CLI | 9.9868 | EXIT0 |
| 其中安装后的依赖console | 2.2448 | EXIT0 |

16模块在源码、wheel及安装目标的SHA相同；配方22键按具名后端与DAG选项校验。TF32设置保持调用者状态，CUDA初始化状态始终为False；这不是GPU前向的精度记录。NVML导入通过也不证明进程PID可归属或显存合计。

- [offline_install.json](offline_install.json)：完整离线安装工具的实际回执及工具SHA。
- [installed_dependency_entry.json](installed_dependency_entry.json)：另一次实际安装目录的依赖API与 `--require-nvml` console复核。
- [first_failed_preflight.json](first_failed_preflight.json)：先前本地缺tifffile的真实失败；包源码、生产算法和主页依赖声明无变化，依赖补入独立目录后复测。
- [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)：本次1470项包源码的实际冻结清单。

两次wheel分别保存其实际SHA，构建容器的ZIP时间元数据可不同；没有声称wheel逐字节可复现。两次均绑定同一Git源码，关键模块字节一致。所有耗时都是本地安装入口观测，不是影像benchmark；没有脑图或官方重建运行。

## 6. 修改与未完成项

静态审查发现旧工具硬编码配方21键，新输入DAG已有22键。已改成具名语义检查，并复用生产依赖预检和实际安装console。现有科学阶段不修改；历史安装报告保持其原版本。当前整例、GPU前向/实际进程显存、干净Conda环境重建、动态执行/库/文件隔离及整体指标等效均未评估。

## 7. 参考

[Python配方说明](../../../../docs/recon_all/PYTHON_GENETIC_PROFILE.md)、[独立安装说明](../../../../docs/recon_all/PYTHON_RUNTIME_INSTALL.md)、[主页Conda环境](../../../../environment.yml)。该工具属于FNIT包装验证，不含额外算法或原版阶段实现。
