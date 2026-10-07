# Robust 准备态与 M0：SDK 可用性核查

## 当前结果

已在同一服务器的独立 Conda 构建目录找到固定 FreeSurfer d932c45 源码；20 个原源码、头文件及许可证的大小和 SHA-256 与此前绑定一致。Conda 编译器、原 `libutils.a` 和 11 个静态依赖均存在并校验。

当前 Ninja 构建文件没有 robust 的对象声明，10 个已知对象路径也全部缺失。准备态/M0 观察器尚未编译或执行；没有捕获图像缓冲、质心、M0 或 Rsrc/Rtrg。正式精度仍为 **17/20**，原 `relative L2 ≤1e−5` 和支持集零差异门不变。

## 如何解释

SDK 路径已经核实，下一步是独立构建必要 robust 对象，确定准确编译/链接选项与 VNL、header、ABI、动态库身份，再审查新 binary。现有白表面构建选项不能直接充当 robust 编译选项；CMake cache 的配置值也不等于链接成功。

该观察器将使用独立冻结源码参考。即使后续能运行，它也不能称为安装版 binary 的内部 trace。安装版 binary 此次仅核对文件身份，没有执行。既有环境、冻结工作树和运行目录均保持原位。

## 范围与复核

本轮只读取索引、精确源码/编译器/库的身份及构建元数据；新编译、native 执行、MRI 解码或哈希、数组计算、GPU 和生产修改均为0。完整身份与未完成项见 [manifest.public.json](manifest.public.json)。本页不是新的精度或速度 benchmark，也没有新增 API、CLI 或脑图。

原首差仍按[已有只读源码审查](../earliest_branch_readonly_20261006/README.md)的顺序定位：prepared 图像 → Double 质心与 M0 → 首个 A/b。正式候选结果见[保存几何实测](../inverse_real_20261006/README.md)。

## 来源

固定源码为 [FreeSurfer d932c45](https://github.com/freesurfer/freesurfer/tree/d932c45b7941662ea380a05efef580568b98d41a)，相关定义在 `mri_robust_register`。此页只发布身份与状态，不复制 SDK 源码、二进制、影像、atlas、权重、许可证或私密路径。生产运行仍遵守 FNIT 的依赖规则。
