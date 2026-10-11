# 当前 21 项 Python 配方：独立 wheel 安装产物检查

## 1. 功能与范围

本次为已经提交的 `02cb602ee287d07fb0412eea5e97cddba8d02d4e` 构建安装产物，
核对完整 21 项 Python 配方、8 个模块的来源、SimpleITK 固定依赖、CLI 和导入时的 CUDA 状态。
复用项目现有 wheel 检查函数；没有重写重建算法，也没有改变生产默认值。

第一次 v1 从工作树复制，捕获了另一任务正在修改的四个球面相关文件；
其安装检查通过，但不能作为提交版安装门。v2 直接使用捕获 HEAD 的 Git 归档：
1,419 个安装源码文件全部与实际 Git blob 相同，未消费未提交 overlay。
两次真实收据分别保留，不把 v1 改标为 v2。

```mermaid
flowchart LR
  A[捕获 Git HEAD] --> B[归档并校验 1419 文件 SHA]
  B --> C[pip wheel 无依赖构建]
  C --> D[全新 target 安装]
  D --> E[新进程校验 8 模块和 21 选项]
  E --> F[CLI help / CUDA 状态 / wheel 内容]
```

## 2. Python 调用、输入与输出

```python
from pathlib import Path
from run_wheel_checks import run_wheel_checks

installation_report = run_wheel_checks(
    repo=Path("/work/FNIT"),                         # 已核对的仓库，捕获实际 HEAD
    output=Path("/tmp/fnit-wheel-21opts-check"),      # 全新任务目录；不得已经存在
    dependency_path=Path("/work/python-extra-libs"),  # 已安装的 SimpleITK 等补充 Python 库
    git_only=True,                                  # 构建已提交归档，不复制正在变化的工作树
)
```

参数均无物理空间或单位：

| 参数 | 格式与默认值 | 含义与限制 |
| --- | --- | --- |
| `repo` | 必填 `Path` | 实际 FNIT Git 仓库；记录快照前后 HEAD 与逐文件 SHA |
| `output` | 必填 `Path` | 尚不存在的私有任务目录；保留失败日志，不能覆盖旧结果 |
| `dependency_path` | `Path` 或 `None`，默认 `None` | 现有补充依赖目录；本脚本不下载依赖 |
| `git_only` | bool，默认 `False` | v2 必须显式为 `True`；`False` 路径保留实际未提交文件与 Git 差异 |

解释器须已有 pip、setuptools、wheel 和项目声明的依赖。输出是源码快照、wheel、
独立 `installed` target、构建/安装/帮助日志，以及 `SOURCE_BINDING/BUILD/INSTALL/CHECK.private.json`。
返回值为 `CHECK` 字典：包含入口、8 个模块路径/SHA、依赖、21 项配方、CUDA 初始化状态、
wheel 名称/大小/SHA。已有目录、构建、导入、源码 SHA 或参数断言失败会抛异常，保留收据。
没有 MRI 输入、表面输出或脑区统计。

## 3. 命令行

```bash
# 只使用捕获的 Git 提交，记录真实 wheel 与独立安装路径。
python run_wheel_checks.py \
  --repo /work/FNIT \
  --output /tmp/fnit-wheel-21opts-check \
  --dependency-path /work/python-extra-libs \
  --git-only

# 在上述独立 target 的 PYTHONPATH 下新开进程；观察真正 CLI help 的初始化前后状态。
PYTHONPATH=/tmp/fnit-wheel-21opts-check/installed:/work/python-extra-libs \
python audit_installed_cli.py \
  --target /tmp/fnit-wheel-21opts-check/installed \
  --output /tmp/fnit-wheel-21opts-check/CLI_AUDIT.private.json
```

CLI 参数与 Python 相同。`audit_installed_cli.py` 的 `target/output` 均必填；
它调用实际入口的 `--help`，返回码非零、CUDA 初始化或 wheel SHA 不一致会明确失败。
[`export_reports.py`](export_reports.py) 用 `--private-output/--public-output` 脱敏输出；
公开目录必须为新目录，原始/公开 SHA 分开保存，数值、bool 和 null 不改。

## 4. 原软件与编译边界

安装检查没有对应 FreeSurfer 阶段命令，本轮没有运行 MRI 或原版阶段程序。
`setup.py` 只声明 FNIT 的 MSM `_fastpd_native` 扩展，此扩展用于另一功能；
实际 wheel 的唯一 `.so` 是 `fnit/msm/_fastpd_native.cpython-311-x86_64-linux-gnu.so`。
recon-all 的 help 检查没有执行 MSM 算法；没有构建 FreeSurfer 原版阶段程序。
这里只核对安装内容及导入，不等于完整运行的动态库/文件访问审计。

## 5. 当前实测结果与耗时

v2 wheel：`fudan_neuroimaging_toolkit-0.16.0-cp311-cp311-linux_x86_64.whl`，
**5,128,145 字节**，SHA-256：
`59a75d82614c4a6cda50598a2f14e1c07c171dec88600fbca7fd6cdf28b0a4c1`。
构建 Git 归档 SHA：`134a37bf40f46528ad789d2e05ecbda0d0d36fee5b0f71cbbd3983b94f999018`。
wheel 留在私有目录，未发布二进制。

| 项目 | v2 实测 | 范围 |
| --- | --- | --- |
| 归档、展开与逐文件 SHA | 24.765158 s，1,419 文件、Git 差异 0 | 源快照墙钟 |
| `pip wheel` | exit0，6.355518 s | 含项目 MSM 扩展构建，无依赖下载 |
| 独立 target 安装 | exit0，0.802148 s | `--no-deps`，不重装共享环境依赖 |
| 8 模块 SHA / 21 个固定选项 | 全通过 | 独立安装文件与快照一致；未单独记录此检查墙钟 |
| SimpleITK | metadata `==2.5.6`；实际版本 `2.5.6` | 既有临时依赖目录提供 |
| CLI 入口与帮助 | `fnit.recon_all.native_free:main`，退出 0 | 新进程中明确检查当前新增参数 |
| CLI CUDA 初始化前后 | False / False | 不等于零显存测量 |
| 额外 CLI/内容观察 | 4.028278 s | 含 Torch 导入、help、wheel 内容读取，不是重建阶段 |

8 个模块覆盖 Python profile、SimpleITK N4、N4 输入链、主入口、
white.preaparc、最终 white、pial、数字拓扑。21 项参数的**全部键和值**在运行前固定断言，
不是只检查四个后端名。完整结果见 [v2 CHECK](reports/v2/CHECK.json)、
[源码绑定](reports/v2/SOURCE_BINDING.json)、[CLI 观察](reports/v2/CLI_AUDIT.json)。

现有 Conda 环境提供 Torch 2.5.1 等依赖，补充目录提供 SimpleITK/tifffile。
没有创建干净 Conda 环境，没有 GPU 推理、原始 T1 整例、显存峰值、精度或脑图测量；
这些安装时钟不能作为 recon-all 的整例速度。重建质量和脑图见
[本次真实注册回归](../20261010_registration_zero_corner/README.md)。

## 6. 版本与可复现记录

- 历史 14 项配方安装检查保持在[原目录](../20261010_python_wheel_independent/README.md)，不能替代本次 21 项检查。
- v1 真实工作树快照：同一 HEAD 加四个球面 overlay，安装通过但不是提交版产物；
  wheel 5,131,295 B，SHA `82534b9dcacf8a7e63955498bedb318eaabd04bf5879a402315673629f6df7ad`。
  [v1 绑定](reports/v1/SOURCE_BINDING.json)保留四项差异及原始数值；实际脚本为 `run_wheel_checks_v1.py`。
- v2：同一提交的 Git-only 归档，所有安装源码逐项与 Git blob 匹配，全部安装门通过。
  这是产物与导入检查，整体指标等效保持未判定。
- [v1](reports/v1/PUBLIC_EXPORT_MANIFEST.json)与[v2](reports/v2/PUBLIC_EXPORT_MANIFEST.json)
  分别保存私有原始 SHA 与公开字节 SHA，不能混用。所有公开 JSON 保留数值类型与 null。

## 7. 依赖、原实现与参考链接

- [主页依赖](../../../../environment.yml)、[包依赖与入口](../../../../pyproject.toml)、[构建配置](../../../../setup.py)。
- [Python 配方实现](../../../../src/fnit/recon_all/python_profile.py)。
- [独立 SimpleITK N4](../../../../docs/recon_all/N4_SIMPLEITK.md)。
- [完整 Python 流程当前状态](../../../../docs/recon_all/PYTHON_SURFACE_PIPELINE.md)。
- [SimpleITK 项目](https://simpleitk.org/)；这里仅检查安装版本，真实 N4 算法对照见其功能页。
