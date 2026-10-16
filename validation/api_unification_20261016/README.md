# 0.17.0 公共接口统一验收

日期：2026-10-16；对照基线：`97b708884180b2c9894689fc369610dfd2ba3588`。

## 1. 交付内容

- 用户指定的33个功能提供同名 `FNIT_` Python入口和console脚本，也支持 `fnit FNIT_<name> ...`。
- 正确拼写为 `FNIT_convert_space`、`FNIT_fMRISurface_pipeline`。没有发布错误拼写的别名。
- 直接复用成熟模型类／文件函数，配置和结果类型保留；旧计算入口仍支持显式导入。
- `FNIT_newmsm(method="msmsulc"|"msmall")`、`FNIT_noddi(method="amico"|"classic")` 合并方法选择。
- `FNIT_mshbm(input_kind="profiles"|"timeseries"|"volume")` 明确已有三类输入。
- 两项独立准备helper从生产模块和安装包撤下；历史实验从固定SHA快照加载。
- 主入口、细分功能页、签名清单与迁移说明统一，旧benchmark保持原producer身份。

完整输入、输出、参数与示例见[API索引](../../docs/API_REFERENCE.md)、
[方法选项](../../docs/API_METHODS.md)及[完整签名](../../docs/API_SIGNATURES.json)。

## 2. 代码与数值路径

[源码审计](source_audit.public.json)检查了相对基线发生变化的39个既有生产Python文件。
剔除docstring和注释后，已有函数定义只发生6处入口变化：根CLI的`main`与5个lazy导出`__getattr__`。
在这些修改文件中另有41个既有函数定义未变；明确撤下的准备模块单独记录。
新文件用于注册、方法分派或CLI透传，不引入新的配准、分割、追踪或统计算法。

对象identity、签名和默认值由测试逐项对照；NODDI继承原`__call__`/`run`，
MSM与MS-HBM透传原配置、QC、迭代预算和输入对象。MSM还验证了全局选项位于方法名之前的旧用法，
以及配置/输出值恰为方法名时不误识。

这项审计证明本轮接口改动的范围，不代替完整MRI数值验证。本轮没有重跑真实MRI benchmark，
不报告新的精度、速度或显存结论；既有真实输入、分步骤测量和脑图继续在[benchmark索引](../../docs/BENCHMARK_INDEX.md)查阅。

## 3. 验证记录

| 最终测试组 | 结果 |
|---|---|
| 接口、方法分派及相关API | 324 passed |
| 连接组与空间处理组件 | 1475 passed；369 subtests passed；3 skipped |
| 独立wheel | 39项导出、33项CLI help、5项实际console调用通过 |

不同测试组按各自范围报告，不将重叠测试相加。测试运行时间不作为MRI pipeline耗时。

最终机器可读结果见[测试记录](test_receipt.public.json)和[独立安装检查](install_receipt.public.json)。
安装检查从独立target目录导入wheel，覆盖33个主入口、39个根导出、33项CLI help及5个实际console子进程。
小型NIfTI smoke仅检查文件合同：原fslmaths函数、FNIT别名及新console的数组、空间affine及gzip文件字节相同。
该输入不是MRI benchmark，不用于方法精度声明。新增运行时依赖为0。

本轮组件回归还修复了基准准备工具的主机核验位置：跨主机只读配置／冻结资源核验不要求调用者位于CPU执行主机；真正preflight和recon执行前仍强制匹配声明主机。被动核验不启动CPU任务，错误主机的执行仍拒绝。

### 原有测试合同修正

本轮发现两项原有测试与基线的已发布行为不一致，修正测试而不改数值代码：

1. `test_public_api.py`仍断言SynthMorph `apply_transform()`不接受`device`；基线已有`device="cpu"`和`frame_chunk_size`。
2. `test_tf32_defaults.py`仍要求SynthSegParc构造时改变全局TF32开关；基线已使用前向作用域并恢复调用方设置。
   在原基线上单独运行可复现同一失败。本轮将该类纳入已有作用域保持测试，同时验证调用前True/False。

### 全库测试边界

直接收集全部tests时有14项既有错误，均在原基线的对应测试中复现：
3项同名测试模块冲突，3项本地环境缺少已声明的`tifffile`，1项旧相对导入问题，
7项仍导入已不存在的MSM `_fastpd_native`。因此没有声称全库测试全部通过。
本轮入口回归、相关组件测试及隐私审计按各自实际结果记录，不把未运行或收集失败计为通过。

## 4. 隐私与证据保留

[隐私验收](../documentation/public_api_20261016/README.md)覆盖当前公共树、文件名、
PNG文本元数据、gzip、profile与证据归档内部；原始与发布文件SHA分开记录。
科学数值、输入/输出/源码SHA、原作者归属和许可证保持。
整理当前树不等于清除Git历史，本轮使用普通提交，不重写历史。

最后一轮[全量发布扫描](publication_audit.public.json)包含环境身份、私有路径／地址与凭据形态，
实查18,658份候选文件，0命中；58份无文本元数据的二进制单列，不声称扫描了其压缩像素字节。
扫描规则保存在仓库外，结果不含规则或匹配正文。

## 5. 版本与参考

| 版本 | 本轮关系 |
|---|---|
| 0.17.0 | 统一命名、方法选择、console、文档及隐私发布层。 |
| 0.16.x | 数值实现和真实benchmark的既有来源；各结果仍对应实际生成版本。 |

对应原软件命令、原实现仓库与论文继续在[33项功能映射](../../docs/API_REFERENCE.md#4-原软件对应关系)及其功能页维护。
