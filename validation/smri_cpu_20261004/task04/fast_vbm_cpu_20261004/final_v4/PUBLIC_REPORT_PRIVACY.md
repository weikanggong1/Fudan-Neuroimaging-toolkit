# v4 JSON 公开报告检查

`build_v4_report.py` 的 `ensure_public_report(serialized: str) -> None` 在写出公开报告前检查已序列化 JSON。它使用通用账号模式和 RFC1918 私网 IPv4 范围检测，避免在守卫里保存具体服务器账号或地址。既有私密路径、许可证、环境和命令标记检查保留。

输入 `serialized` 是调用方通过 `json.dumps` 生成的纯 JSON 字符串。成功返回 `None`，发现禁止信息则抛出 `ValueError`；错误信息不回显被检查的账号或地址。函数只检查文本，不改写 JSON、指标或已有报告，不加载 MRI，因而不涉及影像空间、单位、CPU/GPU计算或原软件独立 CLI。

```python
import json
from build_v4_report import ensure_public_report

anonymous_report = {  # 已匿名化的报告内容
    "status": "complete",
    "wall_seconds": 12.5,
    "source": "reviewed-anonymous-source",
}
serialized_report = json.dumps(anonymous_report, ensure_ascii=False, allow_nan=False)
ensure_public_report(serialized=serialized_report)  # 通过后方可保存原字符串
```

原报告生成 CLI 参数保持 `--collected`、`--prior-public` 和 `--output`；没有新增影像算法或原软件调用。已有 v4 数值记录保持原文件，检查通过不会重新计算或改变它们。六项标准库测试覆盖匿名报告不变、通用账号、三个 RFC1918 范围、公开地址及版本号、既有路径/许可证/命令守卫，示例账号和地址均为虚构测试内容。

```bash
python -m unittest test_build_v4_report_privacy -v
```
