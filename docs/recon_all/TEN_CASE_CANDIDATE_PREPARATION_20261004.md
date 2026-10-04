# 十例精度候选准备（2026-10-04）

## 版本与范围

候选工作分支以创建时真实 `origin/main` 的 `cc9402734faeba93b3a13c29932fa1392eaccf62` 为固定基线。合入启动补丁 `5f75ed5c` 后的冲突解决提交为 `9f7ed77308224ef1bd5d5c7d3eb89cd648a1c949`，五个精度文件合入后的源码提交为 `127eb30656ae60a639e44634b642bbb945bc93d1`。随后本页仅补充候选的证据说明。

保留 main 的完整父进程 CUDA cache 释放函数与单次调用、设备解析、取消逻辑及诊断测试。启动 worker 的 AST 与已验证 5f 版本一致，加入 READY/GO 屏障和可信的 pre-callable bootstrap OOM 有限 fresh-exec 重试。main 的 `native_free.py`、`segstats_wmparc_python.py`、`volmask_python.py`、`mris_register_average_numba.py` 保持逐文件字节一致。原生产 API/CLI 不变。

## 五个精度文件的真实阶段依据

| 文件 | 实际源文件 SHA-256 | 同输入证据及范围 |
|---|---|---|
| `input_talairach_chain.py` | `ab0a02eca9a738cf9c43e0cbcf179165fe187529bbc34ad1929b619fd60bc8d3` | `task_02/synthstrip_writer.json`：旧两例冻结真实网络输出的 ABBA writer 回放，值和几何差异 0；输出恢复 uint8；非整数输入明确拒绝。未验证当前 main 的完整 Talairach 输入链。 |
| `ca_normalize_python.py` | `167be9d72496b7279b98e52b8f0d6c0808d2c09d02df24af1fed0fd331100b9b` | `task_03/production_fix_abba_completed.json`：旧两例冻结输入 ABBA 8 次调用，候选 norm/ctrl 与官方差异 0。原 driver 未记录退出码，sub-02 性能稳定性未确认。 |
| `place_pial_python.py` | `2377f847610a935859935e7ecafdbbd52a74bd4e27ad372a8324b49249d18e33` | `task_05/placement_complete_v1/completion_summary.json`：一例真实输入双侧完整 pial 四轮，11 个输入未变，有序面未变，mean/P99/max 位移均 0；该阶段集合 14 complete、0 failed。未采集同刻进程树显存峰。 |
| `surface_stats_cache.py` | `6f64b40773c61d46fd167c34a74530d9475dd7992b6607c99df14511826ab2a1` | `task_05/roi_lh_v1.json`：历史 LH 真实相同 surface/pial/thickness/annotation，threads=2、cuda:0，ABBA，候选相同 face-area oracle P99/max=0。 |
| `surface_roi_gpu.py` | `5842a1a57ab0983b5735d3631826e9b89bd7123c271d18a4e4ad1161fb0ee758` | 本文件仅 docstring 修改；`task_05/first_phase.json` 绑定 SHA，当时的 queued 不能称为已完成测试。其 cache/ROI 行为的后续完成证据为上行 `roi_lh_v1.json`。 |

以上是旧 `816e5610417a4c587caf321049438a9554139016` 冻结输入的阶段诊断，五个候选文件 SHA 与各回执准确对应。阶段依赖及完整调用链没有因合入新 main 自动获得验证。CA 历史 README 保留诊断参考作用。新的十例候选连续链尚未运行，整体官方等效为 `not_assessed`；既有 8f 的整例和官方 18 phase 结果保持原版本绑定。

## CPU 测试与后续验证

原 Conda Python 隐藏 CUDA 运行启动调度标准库测试 30/30 与 CA 坐标测试 3/3。main 17 个测试与 5f 28 个测试的函数名并集为 30 个，候选全部保留。原 Conda 缺 pytest，独立 `--system-site-packages` 测试 venv 安装固定 pytest 8.3.5；pial/cache 使用原始 pytest fixtures，四套原始测试最终 40 passed、exit=0、pytest 内部 43.81 秒（外部 44.948272 秒）。结果与轮子、程序、源码完整 SHA 保存于独立运行回执，不能把这些人工小输入单测称为真实 benchmark。

独立 CPU workspace：`FNIT/workspaces/recon_accuracy_20261003/ten_case_candidate_cpu_v1`；运行回执：`FNIT/runs/recon_accuracy_20261003/ten_case_candidate_cpu_v1`。复现聚焦测试：

```bash
# 仅 CPU 测试；环境复用原 Conda 包，不启动生产整例。
candidate_workspace=/cwStorage/home/gongwk/Notebook_code/FNIT/workspaces/recon_accuracy_20261003/ten_case_candidate_cpu_v1
cd "$candidate_workspace"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$candidate_workspace/src" \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$candidate_workspace/test_venv/bin/python" -m pytest -q \
  tests/recon_all/test_hemisphere_parallel.py tests/recon_all/test_ca_coordinate_contract.py \
  tests/recon_all/test_pial_rejected_stop.py tests/recon_all/test_surface_stats_cache.py
```

后续由协调者复用已有真实检查点验证当前候选阶段，再从原 T1 空目录验证新候选；准备阶段没有运行生产、推送 main 或修改冻结 8f 源码。原函数的详细输入、输出、参数、原软件调用与参考文献见各模块功能页；本页仅记录版本整合和证据范围。
