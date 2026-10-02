# 私有安装验证：最终 wheel 和部分原生 bundle 已通过

最终算法 `8d750e2` 的实际 wheel/target 安装和私有原生 bundle 已在 gpucw1 通过。原生 bundle 于 UTC 2026-10-02 16:21:18 完成，状态为 `passed`，attempt 为 `reuse_white_bundle_v2`。本轮重编 GCA；white 复用已通过真实阶段回归的 FNIT 独立源码构建；其余 13 个程序复用原独立 Conda 构建，SHA 不变。

## 1. 最终 wheel

协调者另经安全审批部署固定 `8d750e25d4d067a43edb788a96b2086a1c031ba0` 的必要运行源码到 `candidate_8d750e2`，归档 4,247,792 字节，SHA 为 `45cb2a8260883f58c5640af82b0903ed9d6f3339936730385dc198434d7e88c9`。归档不含 examples、validation、原始 T1、权重和个人许可证；协调者核验了 1487 个文件 SHA。

实际 `pip wheel --no-deps --no-build-isolation`、`pip install --target`、recon CLI 和七项 API 导入通过，总计 30.677 秒，UTC 14:00:37 完成。当前 MNI chain/sampling/inverse、normalization CUDA、native runtime selection 和主 recon API 均从安装目录导入。171 个 `installed/fnit/recon_all/*.py` 与冻结源文件 SHA 完全相同；GCA helper/header 在 source、wheel、installed 三方 SHA 相同，FastPD 扩展实际编译并安装。MRI 候选的 Python 输入目录是该实际 `installed/`。

- wheel SHA：`a8297b9b8168ac496b3c9708e16109b3951681e08aa3f87794794481287b474d`。
- FastPD 扩展 SHA：`47d92c64de0a3a96eaa04c39fd9656ff51d5326142031aa6e3b76f22e897823e`。
- [实际安装报告](final_wheel_8d750e2/report.json)、[源文件与成员校验](final_wheel_8d750e2/installed-source-check.json)、[最终 wheel 状态](final_wheel_queue.json)、[启动记录](final_8d750e2_launch.log)。

## 2. 原生输入与实际操作

最初 supervisor `45414` 分阶段持共用锁 `/tmp/fnit-recon-five-20261002-gongwk.gpu.lock`，CPU 线程预算 4，禁用 CUDA。source、旧 `1dd6df8` wheel 和 GCA 阶段实际通过；旧 wheel 单独绑定 `1dd6df8fc10d8446491fea11bdf3dadb507428f9`，没有改标成最终 wheel。GCA 有限重编及能力查询于 UTC 15:41:47 完成。见 [阶段状态](native_partial_v2/status.json)、[GCA 构建报告](native_partial_v2/gca_build/build.json)、[旧 wheel 报告](native_partial_v2/wheel_install/report.json)。

原始固定 FreeSurfer 包实测 SHA 为 `2e76f40415f3e334b6fcd2ce548b451e9219bc9ffaecc46e752d4853e13aff0a`，全量原始源码 tree 在锁内重新核验，SHA 为 `313afb62ea5b5c7d5aa9d78659403b126c63e7cdd91465391ce6a2138c93693c`。见 [实际来源](source_validation_1dd6/source-provenance.json)。GCA 的共享 Conda source/build 只作只读输入，修改后的源码、对象和程序写入私有 `gca_build/`。

提交原队列时，五份 helper 的服务器实测 SHA 与冻结 `24390dc` 相同，见 [report.json](report.json)。随后协调者修复 white helper 相对 `ld_map.txt` 的输出副作用：解析绝对路径并将 map 写到候选目录；算法和编译参数不变，原 build 未写入的 mock 回归通过。原队列已准备仅重定向 map 的兼容 argv，但该 white 编译没有执行。

white 实际复用 `task_02/native_build_v3/candidate/mris_place_surface_fnit_hotspot`。重新核验固定 d932 六份源码、派生有限补丁、对象、builder SHA、Conda 编译器、完整编译/链接 argv 和能力；该二进制与 `native_three_stage_v1` 实际测试产物 SHA 相同。实际原 source/build 位于 Conda prefix 的 `share/fnit/recon_all_native_full/`，原链接 archive SHA `e682f769892616f4cdbe35168017808bad4eb7b2aab8c802c120eca87184d8ad` 已重新匹配。另一个 codeload-probe build archive 的 SHA 为 `2fd873c6f4fb7b836a3207ef3e1356eba3d5d5a254ddbe49de04fb7b80f6f7ab`，属于不同构建输入；没有将它冒称为 task02 的原链接输入。见 [white 复用来源](native_partial_v2/white-reuse-provenance.json)。

停止前检查原 supervisor 和其 white flock 处于自有排队状态、没有计算子进程、没有 white 构建目录；暂停后再检查，安全终止两个自有进程。旧 white 阶段明确记录为 `superseded_not_run`，没有伪记成重编通过。新 attempt PID `11425` 只复制自建产物、调整私有 RPATH、查询能力和核验哈希，没有启动影像、GPU 或性能计时。见 [被替代的排队记录](native_partial_v2/superseded-white-recompile.json)。

## 3. 安装产物

实际 bin：`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/parallel_20261002/private_install_v1/native_bundle/bin`。

| 程序 | 本轮操作 | RPATH 前 SHA | 安装后 SHA |
| --- | --- | --- | --- |
| `mri_em_register` | 实际有限重编；capability 2、完整优化器、`upstream_ROMP_partials` | `c371185fb08a583cae663af5acb0a7d56f96e4d3cdd18d6ff3c389b94b2b02ac` | `70911f6e3ed8f240f2fc745ad1060664c893da47ff3cb6d591d219760f59079f` |
| `mris_place_surface_white_fast` | 复用 task02 独立固定源码候选；capability 1；仅用于两项 white | `88b09e3cff560e2ef09cddf16213a34540b09f72c7b66723da0de1fa5c9ab1db` | `c99fd5ffdaa6c65219272cd94211fd45c59fafc19b8750917e53bae29a7d6e02` |
| 原 `mris_place_surface` | 复用原自建程序；保留 pial 与 surface metrics | `44ad3994f8b86809b447919094273d3aa591b49be935ac4e09139e66bdddfe65` | 同左 |

两个安装候选的 RPATH 均为 `$ORIGIN/../lib`，缺库检查和安装后的能力查询通过。另外 13 个程序包括原 pial，从 `serial_20261001/native_bundle` 复用，均与 baseline manifest SHA 相同，没有复制系统预装 FS 二进制。白程序明确记录 `rebuilt=false`、`reused_kind=task02_independent_fixed_source_conda`、`rpath_adjusted=true`；GCA 为 `rebuilt=true`。

15 项程序路径、安装 SHA、来源和能力见 [安装 manifest](native_partial_v2/installed-native-optimizations.json)，其 SHA 为 `321f1563264b649dcd4d8daf1120737a01d7995b2b6aebac1dea9b7f8482c856`；[实际 bundle 校验](native_partial_v2/bundle-validation.json) 与 [执行日志](native_partial_v2/reuse-white-bundle.log.json) 已回收。个人运行许可证仅记录路径，不读取内容；共享环境、source/build 不修改。

## 4. 验收边界

这是实际通过的私有部分安装。完整新 Conda 创建、完整 `tools/setup_recon_all_native_conda.sh` 正向安装、当前全部 15 个生产程序从头构建和无预装软件的物理隔离尚未执行；只实测 setup 缺 Conda 工具链的失败门禁。普通 wheel 和 sdist 当前都不能代替含 shell/N4 CMake/C++ 的完整 checkout。

安装成功、171 个 Python SHA 精确和能力查询不等于完整 MRI 对照通过。阶段性能继续绑定既有真实 GCA/white/pial 报告，完整两例候选须在 gpucw1 同硬件、共用锁和四线程预算下另行运行，不能相加阶段收益。

最初完整源码归档上传被自动审批拒绝，未执行；旧私有路径改用服务器内既有 `1dd6df8` 部署，没有绕过拒绝上传。最终必要 `8d750e2` 源码归档是协调者另外审批和部署的独立路径，来源见第 1 节。

CPU 替代尝试只读复用 gpucw1 和已认证 headcw ControlMaster；两条路径对 `nodecw10`、`nodecw12` 新 BatchMode exec 均拒绝认证（退出 255）。未进入 CPU 节点，环境兼容性未测，没有申请新密码、读取凭据或增加登录节点计算负载；按协调者指令停止扩展登录检查。见 [CPU 替代记录](cpu_host_alternative.json)。
