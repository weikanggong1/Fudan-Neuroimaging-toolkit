# 完整EM合并后的安装与入口核验

## 1. 功能与范围

冻结Git `342aa4dd05fcfd436ae9dcf6b0991f0f5984734b` 的已提交源码，离线构建wheel、安装到新目录，再核验完整EM、可选拓扑GPU距离接口、生产配方和实际console。没有运行MRI、GPU/JIT或原软件。

## 2. 输入、输出及Python调用

输入为[1386项安装源码与元数据清单](SOURCE_MANIFEST.json)的相对路径、字节数和SHA-256，没有工作树overlay。清单SHA为 `febb12a228cd720b7bfe688279eef4e0ff1b4c3db3bde9d89a1847637e9f57f4`。从指定Git提交导出清单所列文件后，复用工具的四项必填Path参数，输出新wheel、独立安装目标、入口及总报告；无影像shape、坐标空间或单位。

```python
from pathlib import Path
from verify_recon_genetic_profile_install import verify  # tools目录加入PYTHONPATH

installation_report = verify(
    source_directory=Path("/data/fnit-frozen"),  # 已冻结src与项目安装元数据
    source_manifest=Path("/data/source-manifest.json"),  # files记录bytes/sha256，git_base绑定提交
    output_directory=Path("/data/new-install-check"),  # 必须不存在且位于源码目录之外
    python_executable=Path("/conda/envs/fnit/bin/python"),  # 已装主页依赖、setuptools和wheel
)
```

校验、构建、导入或console失败会抛异常并保存私有日志，不自动下载依赖、不覆盖历史报告。完整参数与返回结构见[配方说明](../../../../docs/recon_all/PYTHON_GENETIC_PROFILE.md)。

## 3. 命令行

```bash
python tools/verify_recon_genetic_profile_install.py \
  --source-directory /data/fnit-frozen \
  --source-manifest /data/source-manifest.json \
  --output-directory /data/new-install-check \
  --python-executable /conda/envs/fnit/bin/python
```

构建与安装使用no-index/no-deps，在已有依赖环境中执行；不是新建干净Conda环境测试。

## 4. 原软件对应

本工具只有安装和接口验证，无独立FreeSurfer命令。算法参考与真实数据对照见对应EM、拓扑及体积前段说明。

## 5. 实测结果

| 操作 | 墙钟/秒 | 结果 |
|---|---:|---|
| 构建wheel | 2.400593 | EXIT0 |
| 独立安装目录 | 0.766072 | EXIT0 |
| 导入、配方、依赖及三项fresh CLI | 10.241632 | EXIT0 |
| 其中生产CLI help | 1.944946 | EXIT0 |
| 其中GA文件CLI help | 2.311512 | EXIT0 |
| 其中安装后的依赖console | 2.233504 | EXIT0 |

24模块的source/wheel/installed SHA均相同。wheel大小5,307,871 bytes，SHA为 `e52c9220dd221bf0f73c26268e79dbee93d757b1e05a1633f32ac091ac8dd064`。实际调用选择器声明完整stable/all-sample EM已接入，但GA/pial与整例质量仍待验证；公共默认未切换。

导入前后CUDA均未初始化，matmul TF32=False、cuDNN TF32=True，完全保留调用方设置；这不是GPU前向精度记录。SimpleITK2.5.6及扩展库身份核验通过，NVML只作为可选依赖导入，不查询驱动。GPU显存、真实MRI、脑图与整例隔离均not_assessed。

[完整机器报告](report.public.json)保留源码/wheel/安装哈希、实际依赖版本和计时；[根回归收据](root_receipt.public.json)记录27项EM/API/入口测试加12项子测试，以及24项NVML故障和3项WM语义合同。合同小数组与mock不代替真实MRI；另有两例完整EM/CA与四条原始T1→filled链的实际证据。

## 6. 更新记录

原16模块安装回归仍对应 `e9e89cac`。本轮完整EM与可选GPU距离核合并后，新增具名接口核验并重新冻结、构建和安装，不能复用旧wheel作为新版结果。科学算法与GPU默认策略没有因安装核验而改变。

## 7. 参考与原代码

- [完整EM](../../../../docs/recon_all/GCA_COMPLETE_PYTHON.md)
- [体积前段](../../../../docs/recon_all/N4_VOLUME_PREFIX_PYTHON.md)
- [主页Conda环境](../../../../environment.yml)
- [验证工具](../../../../tools/verify_recon_genetic_profile_install.py)
