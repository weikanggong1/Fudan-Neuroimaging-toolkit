# 当前独立 Python 配方与 N4 选择：证据范围

本目录新增只读当前调用链审核和选择接口契约。53 个源码/安装文件重新绑定 SHA，24 组功能参考函数重新解析；旧静态审核仅复用检查清单，其旧哈希和结果保留，不作为本次源码的证据。

- `capture_v1/audit.json`：实际 `5cc98494` 加工作树捕获、默认/显式 N4 的 21 键字典、所选与未选运行路径、历史报告哈希及边界。
- `contracts.json`：聚焦接口测试通过；扩大 SimpleITK 测试因本地缺库的一项失败完整保留。测试未运行真实 N4。
- `generic_libraries.json`：启动审计下通用 Python 包实际导入；SimpleITK 本地未安装。
- `case_summary_v2.json`、`verify_case_mapping.py`：从实际病例路径、输入和原报告 SHA 派生映射收据；更正中文表的 SimpleITK sub-06/sub-07 颠倒为 99/28，Torch 4016/3259 和 filled 4671/2291 顺序保持；原机器报告不修改。
- `build_audit.py`：具名 `--repository` 和新 `--output` 路径；源码或 HEAD 在捕获期间变更时拒绝发布，不覆盖原报告。
- [七节中文说明](../../../../docs/recon_all/PYTHON_PROFILE_N4_SELECTION.md)：参数、失败行为、完整示例、参考实现和既有真实数据范围。

```bash
PYTHONPATH=src python \
  validation/recon_all/optimizations/20261010_python_profile_n4_selection/build_audit.py \
  --repository . \
  --output /tmp/fnit-profile-n4-audit-new
```

这是调度/依赖审核，不是新的原始 T1 整例或物理隔离验收。完整 Torch N4 的量化与下游误差不隐藏；整体等效 `not_assessed`。共享单例、batch、benchmark 接线由协调者另行测试；此处没有修改这些文件，也不把缺少原版调用等同于解剖结果通过。
