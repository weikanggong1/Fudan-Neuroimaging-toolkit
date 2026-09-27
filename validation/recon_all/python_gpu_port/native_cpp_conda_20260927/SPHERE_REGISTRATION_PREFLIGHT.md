# Conda C++ 球面及配准预检（2026-09-27）

在 gpucw1，FreeSurfer 8.2 源码提交 `d932c45b7941662ea380a05efef580568b98d41a` 构建的 `mris_inflate`、`mris_sphere`、`mris_place_surface`、`mris_register` 使用同一候选被试的左半球输入顺序运行，退出码为 0。此预检只确认程序及外部数据可独立执行，不是与官方输出一致性比较。输入几何来自此前近似的 SynthSeg filled/拓扑修复路径。

- Conda 环境：`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927/conda_env`
- 运行和原始日志：`/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927/sphere_registration_preflight/{run.sh,run.log,status}`
- 外部 `FS_LICENSE` 只作为环境变量提供；`FREESURFER_HOME` 指向 15 项纯数据资产目录。
- `mris_inflate` 原生日志：0.2 分钟；`mris_sphere`：0.0871 小时；`mris_register`：0.0783 小时。此为单次负载条件下的命令日志，不用于声称稳定加速。
- 输出哈希：`lh.inflated` `3f0830c3ac3c1da27c9c71c3165aa825358e7a11fe22263b8056391395bb50e7`；`lh.sulc` `0a114093b620fd6cd82d2aa4f48a0aa63c720bdcd4960751e74856b574f6735b`；`lh.sphere` `a4ea4122d588776b70bc41ecc9acfcc94dcc84bfa8770ab5dfc45c7f5487f694`；`lh.sphere.reg` `d6acab70e2f65115e454c93a568e7ce49d9e24b81ffe2cf507be1a2f75c5f606`。
