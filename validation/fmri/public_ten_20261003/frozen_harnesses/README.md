# 冻结的参考检查脚本

[清单](manifest.public.json) 保存本轮真实旧 harness 和后验检查的字节数、SHA 与使用范围。新的默认入口位于上一级；这些文件用于核对历史报告的实际源码，不能因默认入口升级而改写旧报告。这里仅保存 FNIT 自有 benchmark helper，不发布原 MRI 软件代码。

- `run_reference_cohort_v1.py`：CON01 成功的 attempt-02 原完整 MRI harness。
- `run_reference_cohort_v2.py`：CON03–07 和单独 gpucw1 控制的原 harness。
- `recover_reference_saved_v1.py`：前四例 corrected 的实际补检源码；v2 为此前未产出修正结果的等待版本，v3 是 CON06–11 两种原 QC 状态的实际补检源码。十例现均已保存，原冻结字节不变。
- 两个 `normalize_reference_report*` 是不同字节的未用原型，不是 corrected 报告的生成程序。
- `collect_reference_stages_v1.py`：此前四例参考与单例硬件控制汇总的实际源码。它的状态固定为 `partial_cohort`；实际完成数量和全部时间值保留原记录。新的默认入口显式核对预期病例集合。
- `verify_single_voxel_stats_real_v1.py`：CON04 四次完整旧/新 FNIT writer 的实际 producer。v2 是此前严格警告/退出码的本地中间版本。新默认 v3 在任何 mkdir 前拒绝与 MRI/source/raw/report 互相包含或已存在的输出，严格匹配原警告，失败合同退出码为 1；旧真实报告与远端实际源保持不变。

旧 recovery 脚本通过 `PYTHONPATH` 导入上一级固定 SHA 的 `validate_reference_saved.py`。公开默认 `recover_reference_saved.py` 优先核对当前 attempt 的 `launcher.snapshot.py`；缺少 snapshot 的早期 legacy attempt 可用 `--source-root` 指向本目录，按原报告 SHA 选择冻结 harness。同 SHA 的重复别名不会构成多份独立源码。已存在 corrected 或私有绑定时拒绝覆盖，应读取原绑定结果。

`audit_sphere_orientation_v1.py` 是独立诊断的 HCP 路径前置失败版本；v2 因新 Workbench provenance 导致完整 rotated GIFTI SHA 不匹配而严格拒绝。两者不产生 MRI 结果。默认 v3 只接受原 native/sulc/affine/reference SHA 匹配，明确保留 regenerated rotated 文件 SHA 不同和原临时坐标不可逐值证明的边界。

`audit_csf_versions_v1.py` 是原只读计数诊断，因 shape 的 NumPy 整数不能直接保存 JSON 而失败；新默认 v2 显式保存 Python 整数，仅重新读取三份已完成原件，未重跑 MRI 或 PV writer。

`read_saved_msm_absolute_v1.py` 是 CON08 四份最终 sphere 的真实只读 producer，原输入与报告不变；默认 v2 增加对 candidate whole case、reference attempt 和 raw BIDS 根的互相包含拒绝，四项真实 metadata 路径拒绝控制另存，无 MRI 重算。

上一级 `read_saved_registration_qc.py` 另只读摘录 CON10 原 metadata 的 literal QC 字段，绑定实际 `msmsulc.py` 和原完成报告 SHA；其 `folded_solver_faces` 是最后 `_sphere_warp` 插值后的 native 保存前检查，未声称恢复 DATA/control 优化器末态。

`compare_standalone_fs82_reconstruction_v3.py` 是已中断的旧 metadata/stdlb 等待版本；v4 是实际严格等待原 whole complete 的版本，在 CON08 原 reporter failed 后拒绝且未生成数值 attempt。默认 v5 显式绑定独立 late report/files/source/validator；原 a194 的 failed 状态和缺失 API 时钟保持原值。

`run_surface_backend_cold_cpu_diagnostic_v3.py` 是 CON08 原 cold FS8.2 CPU 科学 full API 的实际源。产物返回并通过保存检查后，其 dataclass Path 序列化失败，原 failed 报告保留。默认 cold helper 只修复未来的 Path 序列化；本轮没有重跑 MRI。`compare_CON08_backend_saved_timeseries_v1.py` 是早先部署、未执行数值比较的原 complete-only 版本，实际晚期保存产物对比由上一级 v2 生成；独立 late 检查器和这份比较的当前 SHA 亦列在清单中。
