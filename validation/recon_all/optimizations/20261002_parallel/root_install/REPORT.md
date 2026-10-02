# 私有安装验证：当前排队

2026-10-02，gpucw1 上的 `private_install_v1` 已异步提交，supervisor PID 为 `45414`。各个源码核验、wheel/target 安装、有限 GCA 重编、有限 white 重编和 bundle 校验步骤分别持有同一共用锁，线程总数为 4，禁用 CUDA 设备。当前在等待本轮整例结束后的 CPU 窗口；排队不代表编译或安装通过。

## 来源与范围

- wheel 使用服务器已有 `integration_1dd6df8`，绑定提交 `1dd6df8fc10d8446491fea11bdf3dadb507428f9`。
- 五份原生安装 helper 的服务器实测 SHA 与当前 `24390dc` 相同，见 [report.json](report.json)。
- 原始固定 FreeSurfer 包的实测 SHA 为 `2e76f40415f3e334b6fcd2ce548b451e9219bc9ffaecc46e752d4853e13aff0a`；全量原始源码 tree SHA 将在锁内重新核验。
- 固定 Conda build 只作为只读编译/链接输入；所有修改后的源码、对象、archive 和程序写到新的私有目录。原 white builder 的链接参数含相对 `ld_map.txt`；本次编排明确将其改为私有 `white_build/ld_map.txt`，避免覆盖原 build 的诊断 map。
- 新 GCA 和 white 产物实际重编；其余程序复用 `serial_20261001/native_bundle` 已有独立 Conda 构建，不宣称重新编译。pial 使用原 `mris_place_surface`。
- 个人运行许可证只记录路径，不读取内容。共享 `fnit_main_env`、原始 source/build 不修改。

## 安装验收边界

计划实际执行根 `pip wheel --no-deps --no-build-isolation` 和 `pip install --target`，检查 GCA helper/header、FastPD 扩展、recon CLI/API 实际导入路径，以及两个有限原生候选的能力、RPATH 和程序 SHA。普通 wheel 不包含原生 shell 安装器；原生程序仍从完整仓库 checkout 构建。

完整新 Conda 环境创建、十四个组件从头重编、`tools/setup_recon_all_native_conda.sh` 正向完整安装和无预装软件整例隔离均尚未执行。本轮当前是已排队的私有部分安装验证，性能结果仍引用各自既有真实阶段报告。

完整源码归档上传被自动审批拒绝，未执行；改用服务器内既有、协调者已验证的源码，没有拆分上传或从 GitHub 自动重取。临时编排脚本不含项目源码、影像、凭据或许可证内容。

状态与完整日志保存在服务器 `/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/parallel_20261002/private_install_v1/`。首次实际排队结果见 [launch.log](launch.log)。
