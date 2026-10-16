# 统一公共接口与发布隐私整理

2026-10-16。基线为 `97b708884180b2c9894689fc369610dfd2ba3588`，原树有 18,635 个 tracked 文件，共 1,187,449,051 字节。

## 当前接口与文档

主要处理入口统一从 `fnit` 导入，名称以 `FNIT_` 开头。完整清单、真实参数、模型构造及方法选项见[统一 Python API](../../../docs/API_REFERENCE.md)。功能页保留输入、输出、CLI、原软件对照、真实 benchmark 和许可；[文档导航](../../../docs/README.md)集中提供入口。

模型类及已有文件函数直接复用成熟对象；MSM 的两个模式与 NODDI 的两个方法使用明确选项。配置类型、结果类型及细分算子保留高级兼容接口。已撤下的两项独立亚区对齐准备接口仅保留[历史证据](../../robust_register/target_preparation_20261006/README.md)。

本轮不改变算法、统计定义、精度策略或既有 MRI 处理输出。接口和说明整理没有重跑 MRI benchmark；所有精度、耗时、失败与限制仍绑定实际测量版本。Connectome 索引改为当前已发布的原生追踪报告，历史 PyTorch 追踪结果不再作为当前实现的时钟。

## 隐私与历史文件身份

实际工作路径、连接配置、账号和站点名称改为通用目录、环境标签或调用方参数。文件名中的环境标签同步通用化，并更新引用。cohort 工具的 CPU/GPU 主机由调用者显式指定；local 运行核验配置主机与实际主机相符。

[公开脱敏清单](redaction_manifest.public.json)记录每个发布副本的通用路径、原 Git blob 身份、原始 SHA-256、发布 SHA-256 和字节数。清单不含原私有路径。原始 SHA 对应基线 Git 中的原字节，发布 SHA 对应脱敏后的表示；不能用发布副本声称原始日志或冻结源码的完整文件字节未变。

科学输入、输出和原程序源码 SHA、科学数值、错误记录、计时边界、许可证与原作者归属保持。服务器标签通用化不改变某两次运行是否同机、不同环境或共享负载的事实。历史冻结源码中的路径和说明变更仅属于发布层整理，不能重新标成新入口的实测。

### 已执行的保真检查

| 检查 | 结果 |
|---|---:|
| 脱敏或路径通用化记录 | 5,237 个文件 |
| 文件路径通用化 | 475 个文件 |
| JSON 全部数值叶子与原 Git blob 对照 | 3,841 / 3,841 不变 |
| 历史文件原 SHA-1/SHA-256 token 对照 | 5,061 / 5,061 保留 |
| 压缩 JSON 全部数值叶子 | 7 / 7 不变 |
| 证据归档原有成员 | 123 / 123 保留，另加发布身份清单 |
| 归档内 JSON 全部数值叶子 | 105 / 105 不变 |
| 现存 NIfTI/MGH 头元数据检查 | 13 份，0 个身份匹配，未读取体素 |
| 生产卷积实现去除 docstring 后的 AST | 与基线相同 |
| 图像处理策略 | 仅检查/处理文本元数据，不修改 PNG 像素块 |

保真检查不等于 MRI 方法达到原软件精度；各方法的科学结论仍见功能页。旧日志的完整字节哈希与当前发布文件哈希分开解释。唯一历史证据归档内的 123 个原成员全部保留，其中 91 个文本成员仅脱敏路径和身份，并增加 `publication_redactions.public.json` 区分原始成员与发布副本。原归档的 SHA 继续作为历史原字节身份，当前归档和成员的发布 SHA 见相应清单。

普通提交整理当前代码树，不会清除既有 Git 历史。

## 可重复发布审计

[`tools/audit_public_privacy.py`](../../../tools/audit_public_privacy.py)只读检查当前文件、文件名、PNG 文本元数据、压缩日志、profile 路径和 tar 归档成员。私有规则由仓库外的 JSON 字符串列表提供，不在代码或测试中保存真实身份。输出只含不透明文件标识、文件 SHA 和命中数量，不回显匹配正文或规则。

```bash
python tools/audit_public_privacy.py \
  --repository /data/source/Fudan-Neuroimaging-toolkit \
  --policy /private/publication_patterns.json \
  --include-untracked \
  --output /private/publication_audit.json
```

- `repository`：待发布 Git 工作树，默认当前目录。
- `policy`：必需，仓库外的非空 JSON 正则字符串列表，由数据/环境管理者提供。
- `include-untracked`：检查尚未暂存的新文档和文件；已删除文件不计为当前发布内容。
- `output`：审计 JSON；发现匹配时退出码为 1，否则为 0。

这个审计工具不自动修改文件，也不从压缩像素字节判断文本泄露。泛化的合同测试检查输出不暴露匹配内容、PNG 元数据处理和新增文件覆盖。

本范围[72项工具测试及40个subtests](tool_tests.public.json)通过，具体输入范围的[数值与摘要保真](redaction_verification.public.json)、[重命名行数核对](rename_line_accounting.public.json)及[身份规则扫描](privacy_scan.public.json)可复核。最终凭据形态扫描、测试与安装结果由本轮协调验收记录提供；生产算法的既有实测报告没有被重写成新一轮 benchmark。
