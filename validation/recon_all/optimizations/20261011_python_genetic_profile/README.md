# 完整自有 GA 配方：接线和独立安装合同

## 1. 功能与验证范围

本次把安装包内的完整有序 GA 文件接口接入单例、batch、CLI 和隔离半球 worker。显式 `backend="python", topology_backend="python-ga"` 不被旧数字拓扑配方覆盖；只有真实完成、最终网格质量及三个文件 SHA 均通过才发布标准表面。共享缺陷体积仍在父进程按左、右顺序累计。

这里报告本地合同与离线 wheel 安装。它不包含新的 MRI 算法运行，不代替九缺陷、最终 `orig`、原始 T1 整例或干净环境验收。

## 2. 输入、输出与代码版本

Git 基底是 `bbc6bc0b8a577cacf3cfc303e05242c15e821e66`；四个接线文件作为具名 overlay，未混入其他人的 retessellation 修改。冻结清单 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 包含实际 1462 个源码/包装文件的大小和 SHA-256。安装包中科学 retessellation 源码保持 Git 基底的 `4b20a92f…`，受保护工作区的其他修改不参与该安装产物。

[contracts.json](contracts.json)记录实际合同源码及测试文件 SHA；[contracts.xml](contracts.xml)保留所有结果，仅删除本地主机名。发布函数的输入/输出、编号域、RAS/mm、默认参数、异常和具名中文示例见[完整配方说明](../../../../docs/recon_all/PYTHON_GENETIC_PROFILE.md)，底层文件结构见[GA 文件接口](../../../../docs/recon_all/TOPOLOGY_GENETIC_FILES.md)。

## 3. 复现命令

在具有主页 Conda 依赖的环境中，对同一冻结源码清单运行：

```bash
installation_arguments=(
  --source-directory /data/frozen-source       # 冻结的完整包源码，不是原软件安装目录
  --source-manifest /data/SOURCE_MANIFEST.json # 相对路径、bytes和SHA逐文件核对
  --output-directory /data/new-install-check  # 新空目录，位于冻结源码之外
  --python-executable /conda/envs/fnit/bin/python # 已安装声明依赖的解释器
)
python tools/verify_recon_genetic_profile_install.py "${installation_arguments[@]}"
```

四个参数均必填。工具复制清单中的源码到新构建目录，使用 `--no-deps --no-build-isolation --no-index` 构建，并以 `--no-deps --no-index` 安装到独立目标。随后核对八个关键模块的解码源码 SHA，运行真实 import 和两个 fresh CLI help；不会下载依赖、初始化 CUDA 或执行影像算法。参数及失败行为见功能页。

合同命令使用 `pytest` 对配方、N4选择、batch、发布、拓扑接线及真实 exec 半球目录测试进行具名选择；完整选项在 `contracts.json` 关联的测试文件中。小文件夹具是接口测试，不能当作真实影像 benchmark。

## 4. 对应原软件

本次配方解析、文件 SHA 发布、半球路径审核和 wheel 验证没有 FreeSurfer 独立等价命令。底层科学算法对应固定版本的 `mris_fix_topology -ga`、三轮 `mris_remesh` 及 `mris_remove_intersection`；实际参考程序与种子记录在[参考重复报告](../20261011_topology_native_repeat/README.md)。生产接口不启动这些程序。

## 5. 实测结果与边界

| 项目 | 当前结果 |
| --- | --- |
| 接线合同 | 30 个测试、119 个子测试通过，32 个未选；pytest墙钟27.28秒，JUnit27.260秒 |
| 发布与资源合同 | 全部质量/完成/SHA核对后发布；失败不产生三个标准文件；两半球同名报告保持层级，越侧写入被拒绝；总4线程传为每侧2 |
| 离线 wheel | 5,273,214字节；实际SHA见[installation.json](installation.json)，八模块源码全部匹配冻结清单 |
| 构建 / 安装 | 2.286 / 0.823秒，含完整读写；不是重建时间 |
| 独立入口复核 | 实际模块来自独立安装目标；单例CLI help 1.741秒、GA文件CLI help 2.230秒，含fresh导入 |
| CUDA / 精度 | 导入和配方解析前后状态相同；不初始化CUDA，未测本轮GPU前向或显存 |
| 整例与隔离 | 当前GA配方未完成原始T1整例；干净Conda和无预装软件环境未验证；整体指标等效未判定 |

wheel ZIP 包时间元数据可以改变整体文件 SHA，故每个产物记录自己的 SHA，并额外逐模块核对解码源码。没有以不同 wheel 的整体哈希判断科学实现改变。

真实 MRI 的首缺陷完整评分、状态、随机流和脑图见[面索引完整报告](../20261011_topology_genetic_face_cache/README.md)。九缺陷及最终 orig 正在单独验证；本文不使用阶段图伪装整例结果。

## 6. 更新记录

- 2026-10-11：贯通显式 `python-ga`，传递实际半球线程；避免完整文件接口后重复 remesh/cleanup。
- 保留GA运行目录层级及相对路径/收据，修复双侧同名报告覆盖风险。
- 固定源码基底＋四个 overlay 独立构建安装；新增离线复现工具，不增加主页依赖。

## 7. 原代码与文献

- [FNIT recon-all](https://github.com/weikanggong1/Fudan-Neuroimaging-toolkit/tree/main/src/fnit/recon_all)。
- [固定 FreeSurfer 源码](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)。
- Fischl B et al. Automated manifold surgery: constructing geometrically accurate and topologically correct models of the human cerebral cortex. IEEE Transactions on Medical Imaging, 2001, 20:70–80. [DOI](https://doi.org/10.1109/42.906426)。
