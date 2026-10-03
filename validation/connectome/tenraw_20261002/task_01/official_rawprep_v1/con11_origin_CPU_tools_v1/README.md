# CON11 实际 origin 与独立 CPU 参考流程

origin.json 和 CON11_origin_verified_v2.json 是实际数据核验凭据。v2 guard 对完整体素、affine、原始 AP/PA 与配置、driver、39 个科学源文件进行校验；实际选择为 AP0/PA0。

launch_con11_after_prefix.py 等前九例实际 CPU 完成合同全部核验，且旧控制器没有自身子进程，再在独立 official_CON11_CPU_fresh_origin_v1 目录启动原参考工具的 CPU all-phase：TOPUP、SynthStrip、EDDY。每次只运行一例，8 CPU 线程。未复制 FNIT 场图、mask 或 DWI。

watch_con11_origin_completion.py 核验实际新结果，再与 origin 指向的新 FNIT 目录比较。publish_explicit_case_routes.py 按例发布路径和实际完成文件 SHA256；未完成例保持 waiting，不能据此声称十例已完成。

前九例是恢复自身已完成 CPU TOPUP/SynthStrip 后的新 EDDY，CON11 是新的完整 CPU 运行，计时分别报告，不能伪造恢复 lineage。此目录不修改 FNIT 科学实现，不推送 main。

当前十例完成结果与精简交付见[官方 rawprep 当前说明](../README.md)。本目录的 freeze/route 快照是原启动时的冻结凭据，保留其当时状态；完成时实际路径和 SHA 位于新交付。
