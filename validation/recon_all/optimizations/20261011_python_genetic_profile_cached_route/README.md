# 完整 Python GA：显式面缓存配方与安装回归

## 1. 功能与范围

`backend="python", topology_backend="python-ga"` 的完整 GA 配方现显式启用保留面索引缓存。每缺陷只准备一次不可变索引，动态几何、评分、随机流和最终清理照常执行。文件 API 的默认值仍为 `False`，生产配方显式传 `True`；完成报告与返回参数不同或未实际启用时，在发布标准表面前失败。

本页是接线、发布和安装回归，未运行新的原始 T1 整例。真实阶段报告与本地合同分别列出。

## 2. 输入、输出和版本

Git 基底 `748b86208c5c4e18075321cd13005cdc1fc8e53a`，唯一安装源码 overlay 为 `src/fnit/recon_all/native_free.py`。实际 1462 个包源码的大小和 SHA 见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。八个安装关键模块的源码/解码 wheel 哈希见 [installation.json](installation.json)；受保护工作区 retessellation 修改未混入，科学源码为 Git 基底版本。

[contracts.json](contracts.json)另记录实际测试及库文件 SHA，[contracts.xml](contracts.xml)保留全部结果，仅移除本地主机名。发布文件仍为 surface RAS/mm 的 `premesh`、原域标签和最终 `orig`；输入、参数、输出结构与失败行为见[完整功能说明](../../../../docs/recon_all/PYTHON_GENETIC_PROFILE.md)。

## 3. 复现调用

安装复核使用现有工具，不下载依赖、不修改冻结源码：

```python
from pathlib import Path
from verify_recon_genetic_profile_install import verify  # tools目录放入PYTHONPATH

installation_report = verify(
    source_directory=Path("/data/frozen-source"),  # 与本次清单逐文件一致的包源码
    source_manifest=Path("/data/SOURCE_MANIFEST.json"),  # 相对路径、大小与SHA
    output_directory=Path("/data/new-install-check"),  # 新空输出，位于源码之外
    python_executable=Path("/conda/envs/fnit/bin/python"),  # 已安装主页依赖的解释器
)
```

```bash
installation_arguments=(
  --source-directory /data/frozen-source        # 本次冻结包源码
  --source-manifest /data/SOURCE_MANIFEST.json  # 逐文件清单
  --output-directory /data/new-install-check   # 新输出目录
  --python-executable /conda/envs/fnit/bin/python # 现有Conda依赖
)
python tools/verify_recon_genetic_profile_install.py "${installation_arguments[@]}"
```

全部参数必填。返回报告包含 wheel、代码 SHA、构建/安装/入口墙钟及导入前后 CUDA/精度状态；清单越界、不一致、已有输出或构建/入口失败会抛异常。合同的完整 pytest 参数存于 `contracts.json`。接口夹具不属于真实数据 benchmark。

## 4. 对应原软件

接线与 wheel 发布属于 FNIT，没有独立原软件命令。底层完整 GA、三轮 remesh 和源语义相交清理对应 `mris_fix_topology -ga`、`mris_remesh --remesh --iters 3`、`mris_remove_intersection`；仅隔离 benchmark 启动参考程序，生产配方不调用它们。

## 5. 实测结果

| 范围 | 结果 |
| --- | --- |
| 当前配方/发布合同 | 30个测试、121个子测试通过，32未选；pytest 24.11秒，JUnit 24.087秒 |
| 参数与发布 | 缓存必须为True且与完成报告一致；错误参数不发布三个标准文件；双侧共享输出顺序不变 |
| 当前 wheel | 5,274,180字节，SHA-256 `b33e6999898864d44930e975e7ad344baca83592596dd7aa1a81c7a150d91463` |
| 构建 / 独立安装 | 2.468 / 0.824秒，包含实际读写 |
| fresh CLI | recon-all help 1.686秒；GA文件help 2.213秒；实际模块均来自独立安装目标 |
| accelerator-b现有依赖环境独立安装 | 安装7.137秒、入口组14.299秒，总21.439秒；fresh CLI 2.788/3.501秒；源码、wheel、安装目标八模块SHA相同 |
| 整例包装入口回归 | 1项测试、3项子测试通过，8.84秒；真实main保留python-ga/python及缺省数字配方，在缺失T1预检处停止 |
| CUDA与精度 | 导入/配方解析不初始化CUDA，保留调用者设置；未测GPU前向或显存 |
| 原始T1整例 / 指标等效 / 600秒 | 当前GA配方尚未完成；整体等效未判定，速度目标未达到 |

本次不包含新影像或脑图。真实 T1 阶段及脑图见[当前面缓存库入口](../20261011_topology_genetic_face_cache_wiring/README.md)：六评分的81字段、四NPZ和20,877次随机调用相同，API 48.945→36.928秒，冷CLI 124.854→114.997秒；该对照不是默认完整搜索或整例提速。完整首缺陷见[完整缓存搜索](../20261011_topology_genetic_face_cache/README.md)。不把这些时间相加或外推整例。

已有依赖中的离线安装不证明全新 Conda、无预装脑影像软件环境或当前完整运行隔离通过。

accelerator-b安装报告的安全字段与原始收据SHA见[accelerator-b_installation.json](accelerator-b_installation.json)。独立安装验证的冻结脚本为[verify_wheel.py](verify_wheel.py)和[verify_entry.private.py](verify_entry.private.py)；后者只检查安装目标、21项配方和fresh CLI，无私有内容。前者接受必填 `--workspace`、`--run-directory`：workspace包含本次wheel、与`installation.json`一致的`expected.json`及入口脚本；run-directory必须不存在。只离线安装现有wheel，不构建或下载依赖，验证八模块SHA，返回报告、COMPLETE和私有日志；失败返回非零，不生成成功收据。1核CPU、CUDA不可见，未执行GPU前向或MRI。时间不与本地安装比较速度。

整例包装入口原来只允许native/python且解析时未保留显式GA；已修正。测试源码和JUnit见[benchmark_choice_contracts.json](benchmark_choice_contracts.json)、[benchmark_choice_contracts.xml](benchmark_choice_contracts.xml)，不计入上表原30项接线测试，不改写其结果。

## 6. 更新记录

- 2026-10-11：启用完整配方中的保留面索引缓存，校验实际参数并写入发布收据。
- [首版接线与安装](../20261011_python_genetic_profile/README.md)保持原代码/测试/产物版本；本版单独保存新收据。
- 文件接口增加布尔缓存参数的13项合同通过；新缓存库入口的7项合同和真实六评分对照见其报告。
- accelerator-b的当前wheel独立安装/实际入口检查完成；整例包装支持显式GA并保留选择，新合同单独保存。

## 7. 参考文献和源码

- [FNIT recon-all](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/tree/main/src/fnit/recon_all)。
- [固定 FreeSurfer 源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
- Fischl B et al. Automated manifold surgery. IEEE TMI 2001, 20:70–80。[DOI](https://doi.org/10.1109/42.906426)。
