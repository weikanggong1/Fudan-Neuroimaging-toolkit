# 实际来源读取的两个元数据修复

## 1. 功能

非空输出 ledger 覆盖了 `load_bindings` 中原 map 的 `identity`，导致返回最后一个输出文件的身份。canonical 与 v9 reader 都改用独立 `output_identity/output_path`，原输出 SHA 校验保持。已执行 frozen reader 保留原字节，科学数组与原报告不修改。

另外，两个 mixed CPU caller 依赖 `watch_connectome_actual_completion`，原交付遗漏该 tools 模块。现复用 Task5 实际执行的成熟 helper，SHA 为 `6a9bb156…390cf9`，与远端冻结版本逐字节相同。

## 2. Python、输入输出与参数

```python
from tools.reference import connectome_actual_gpu_origins_v9

origin_map_path = "/shared/actual_origins.json"  # 实际不可变 map，不是输出文件
# cases 是固定原始十例 manifest，options 绑定各个真实来源目录。
actual_bindings, origin_map_identity = connectome_actual_gpu_origins_v9.load_bindings(
    origin_map_path, cases, options
)
```

输入仍为明确 map、十例 case 和原 options；输出为真实逐例绑定及 map 本身的 path/SHA。没有增加科学参数。tiny byte fixtures 只验证 metadata 契约，不作为 MRI benchmark。

## 3. 命令行

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/connectome/test_actual_gpu_origin_bindings.py \
  tests/connectome/test_actual_selected_gpu_origin_reader.py \
  tests/connectome/test_actual_completed_subset_receipt.py
```

## 4. 原软件

此修复不运行 FSL、FreeSurfer 或 MRtrix；它只读取原任务的配置、来源和保存输出。

## 5. 实际验证与时间

headcw 同一 FNIT Python，CUDA 不可见。修复前的新增非空 ledger 测试在两个 reader 中均真实失败；修复后这项通过。after-v1 尚缺 mixed caller archive；v2/v3 暴露仓库遗漏共用 base 模块，原失败输出全部保留。补齐 byte-identical 成熟 base 后，49 tests＋40 subtests 全部通过，pytest 7.08 秒、外层7.418秒，无 MRI/GPU。对应真实十例脑图与320矩阵无损结果见[最终审计](../task_05/final_actual_chain/README.md)。

## 6. 最近版本

2026-10-03：更正 metadata 变量覆盖，补齐实际使用的共用 module，保存 before 和四次 after 的全部原失败/成功报告。`verification.json` 列当前 source、archive、报告 SHA；未用测试成功改写任何原科学计算来源。

## 7. 参考

[完整只读比较与官方步骤](../../../../docs/connectome/actual_cohort_comparison.md)。
