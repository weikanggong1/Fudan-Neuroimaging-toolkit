# 隔离worker守卫合并后的安装与入口核验

## 1. 功能与范围

冻结Git `9292a0a7be63f6bc96f2dfa358261cb1587e0f23` 的已提交源码，离线构建wheel并安装到新目录。核验完整EM、可选GPU距离核、生产配方，以及GCA、WM、Talairach和可选Torch N4子进程的真实入口。没有运行影像、GPU/JIT或原软件。

## 2. 输入、输出及Python调用

输入是[1387项安装源码与元数据清单](SOURCE_MANIFEST.json)中的相对路径、字节数和SHA-256，没有工作树overlay。清单SHA为 `81138d195eae18f341fb7a0485f67e3532fc103a65f9f522c7047549f7ead17a`。本验证没有影像shape、坐标空间或单位；输出是新wheel、独立安装目录、入口报告、完整JSON报告及私有构建日志。

```python
from pathlib import Path
from verify_recon_genetic_profile_install import verify  # tools目录加入PYTHONPATH

installation_report = verify(
    source_directory=Path("/data/fnit-frozen"),  # 清单绑定提交的src与安装元数据
    source_manifest=Path("/data/source-manifest.json"),  # 每项记录字节数与SHA-256
    output_directory=Path("/data/new-install-check"),  # 新目录，须位于源码之外
    python_executable=Path("/conda/envs/fnit/bin/python"),  # 已安装主页依赖的解释器
)
```

四项均为必填Path；路径越界、大小/哈希不同、输出已存在、缺少依赖或子进程失败会抛异常并保留日志，不下载依赖。全部返回结构与限制见[安装接口](../../../../docs/recon_all/PYTHON_GENETIC_PROFILE.md)。

## 3. 命令行

```bash
python tools/verify_recon_genetic_profile_install.py \
  --source-directory /data/fnit-frozen \
  --source-manifest /data/source-manifest.json \
  --output-directory /data/new-install-check \
  --python-executable /conda/envs/fnit/bin/python
```

使用no-index/no-deps构建安装，复用已有依赖；不等于全新Conda环境测试。四个worker的fresh CLI help显式继承生产禁令标志1，在导入和解析边界接受检查，不执行MRI计算。

## 4. 原软件对应

本工具检查安装和接口，无独立FreeSurfer等价命令。守卫语义见[隔离worker说明](../../../../docs/recon_all/WORKER_PROGRAM_GUARD.md)，完整EM与N4算法参考分别见对应功能说明。

## 5. 实测结果

| 操作 | 包含读写的墙钟/秒 | 结果 |
|---|---:|---|
| wheel构建 | 2.293883 | EXIT0 |
| 独立目录安装 | 0.848901 | EXIT0 |
| 导入、配方、依赖及七项fresh入口 | 17.428932 | EXIT0 |
| 生产CLI help | 1.729375 | EXIT0 |
| GA文件CLI help | 2.247405 | EXIT0 |
| GCA worker help，生产禁令1 | 2.024264 | EXIT0 |
| WM worker help，生产禁令1 | 2.047222 | EXIT0 |
| Talairach worker help，生产禁令1 | 1.825281 | EXIT0 |
| Torch N4 worker help，生产禁令1 | 2.020308 | EXIT0 |
| 安装后的依赖console | 2.084441 | EXIT0 |

29模块source/wheel/installed SHA均相同。wheel大小5,309,664 bytes，SHA为 `3e5e2eb0332d328cd92a968a17ae8cc2c6d872a961d103bca4a9d0d65df39e51`。关键模块含完整EM、隔离worker守卫及可选拓扑GPU距离核；公共默认未切换，GA/pial及当前整例质量仍待验证。

导入前后CUDA均未初始化，matmul TF32=False、cuDNN TF32=True，保留调用者设置；这不是GPU前向精度记录。GPU显存、真实MRI、脑图和干净环境隔离均not_assessed。[完整机器报告](report.public.json)保留所有模块/依赖哈希及入口时间；[根回归收据](root_receipt.public.json)绑定36项继承守卫和EM接线合同，均通过，14.10秒。这些CPU合同不代替真实MRI回归。

## 6. 更新记录

[24模块安装核验](../20261011_integrated_python_install_v2/README.md)保留其Git `342aa4dd` 与原wheel。此后GCA、WM、Talairach和可选N4入口继承生产程序禁令，新增fresh worker CLI核验并重新构建安装。本次没有改N4的200轮算法、TF32策略、模型或输出。

## 7. 参考与原代码

- [生产执行配方](../../../../docs/recon_all/EXECUTION_RECIPE_20261011.md)
- [完整EM](../../../../docs/recon_all/GCA_COMPLETE_PYTHON.md)
- [Torch N4 worker](../../../../docs/recon_all/N4_CACHED_WORKER.md)
- [主页Conda环境](../../../../environment.yml)
- [验证工具](../../../../tools/verify_recon_genetic_profile_install.py)
