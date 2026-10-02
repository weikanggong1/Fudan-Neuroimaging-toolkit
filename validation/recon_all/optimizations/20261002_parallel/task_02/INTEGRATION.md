# 接入给协调者与任务1

当前默认生产 Conda white.preaparc、final white 和 pial 不改。三个 wrapper 的签名不变；完整 white 自有候选仍不具备默认替换条件。

完整 `place_pial_t1` 新增兼容 keyword-only 参数：sampling_backend=`cpu`（默认）/`torch`/`triton`；candidate_backend=`tree`（默认）/`snapshot`；device=None（GPU 后端必须给 `cuda:N`）；trace_callback=None（可选每个接受步的诊断回调）。已有调用行为保持。快照候选需要每次新试步重新构建；有 retained-MHT 拒绝状态时保留原 tree 路径。不缓存动态法向或坐标。

实际 GPU 配置必须在子进程启动前固定批准 UUID，并显式传入可见后的逻辑设备；异步接受仍在 CPU。Numba子空间分配沿用原接口；其首轮真实回归逐元素一致。Torch 首轮 GPU 算子虽一致但慢，不选默认；Triton 只在完整四轮与连续链回归通过后讨论调用，不因独立算子快而切生产默认。

安装使用既有 environment.yml 的 `triton==3.1.0`，本次没有新增依赖或工具安装。若协调者支持 environment-recon-all-cpp.yml 的实验 GPU backend，可在该专用环境统一补同版本 Triton；不要从系统预装 CUDA/FS 复制程序。新增公式改写模块应在共享 THIRD_PARTY_NOTICES.md 的 recon_all/FreeSurfer 项下补归属（licenses/FreeSurfer.txt）；本任务只提交专属说明，未编辑共享文件。

专项报告 `summary.json`、`operators_triton/report.json` 与 `monitor.json` 绑定实际 operator commit4674cc2。后续碰撞/完整阶段绑定各自实际提交，不将旧记录改标新版本。保存 CUDA 初始化OOM、Triton编译诊断和GPFS锁失败的独立尝试；修复后计时统一使用协调者的服务器本地锁。根调度、整体138诊断与两例原始T1空目录整例由协调者统一验收。

## 完整产物契约

Python pial 候选只写 `surf/H.pial.T1`，保留面、volume geometry和footer；不写 mrisps、曲率文件或生产日志。现有 `_run_native_pial` 也没有 `--outvol`/`--curv` 参数，检查输出pial网格并写 scripts/H.pial.log。white.preaparc Conda入口写 H.white.preaparc、autodet阈值和 mri/mrisps.wpa.mgz；final white写 H.white和mri/mrisps.white.mgz。`_write_principal_curvature_maps` 等后续阶段另写white.preaparc.H/K等曲率图，任务2没有移动这些调用。候选不能靠有pial网格就冒充完整生产产物；实际原生回归会清点新增/改写文件与hash。Python回调提供诊断不是mrisps输出替代。

## 原生候选（待实测）

`tools/place_surface_hotspots/build_native.py` 按六份入口、计算和哈希支持源码的固定 SHA编译独立control和candidate，复用已授权Conda完整构建的其余对象，不覆盖共享源码/安装。只删除MRISpositionSurface中unused mht_f_current构建；动态MHT和顶点MHT仍每轮重建。若完整三步骤精确且实测有收益，协调者可通过三个已有wrapper的binary参数选择候选；公开安装入口由协调者加入该构建。当前仍选已发布Conda程序，不能仅凭静态未读参数就切默认。没有复制系统预装FS二进制。

独立native_build_v3已成功构建control/candidate；候选 --fnit-placement-capabilities 返回schema_version=1、固定上游提交与skip-unconsumed-repulse-face-table，且uses_cuda=false。构建hash见native_build/control.json与candidate.json。有限install_hotspot.patch已通过git apply --check，新增独立程序名、能力校验、ldd与启动检查，不覆盖原程序；尚未应用到共享安装。

快照首试步回归已完成两例四半球，坐标、动量、次序及retained-MHT重试全部exact；完整首试步比tree略慢，因此不作为默认性能优化。对应collision_snapshot_v1/report.json绑定0d92ef5，未把它标成完整四轮或更新源码验收。完整pial对比包含每步坐标副本、hash、状态和JSON落盘的验证开销；其计时须称为含轨迹验证的完整回归，不能直接用作无诊断生产调用的收益。
