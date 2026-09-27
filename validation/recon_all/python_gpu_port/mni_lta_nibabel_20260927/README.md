# MNI152 体素 LTA 写入替换

这次只替换 `reg.targ_to_invol.lta` 的 Surfa 写入调用。输入是既有的 4×4 完整 MNI152 到被试体素矩阵、完整 MNI152 模板和被试 `orig.mgz`；输出是带源/目标体积几何的 type-0 LTA。完整函数仍由 PyTorch SynthMorph 估计仿射。具体参数与调用方式见 [MNI 辅助链说明](../../../../docs/recon_all/MNI_AUX_CHAIN.md)。

[单函数实测](real_t1_comparison.json)在 headcw 的真实 T1 上固定同一矩阵和两张图像。旧 Surfa 写入与新 nibabel/NumPy 写入的 1,352 字节完全相同，SHA-256 均为 `023be034b9e42f924b1f53c7cb42975525b325ca70620196d6e6aa10c9e0b799`。同一进程交替运行五次，旧写入中位数 0.2181 秒，新写入 0.0328 秒；计时包含读取图像几何和保存 LTA，不包含 SynthMorph 推理。这是小步骤的单机观测，不能用于估算整例速度。

[完整仿射函数复测](full_function_smoke.json)使用先前 MNI 辅助链的同一 `orig.mgz`、权重与模板，在 CPU 上重新运行 `register_mni152_affine`。新旧 `aff.lta` 及 `reg.targ_to_invol.lta` 均逐字节相同；新调用耗时 11.35 秒。这里的“旧”指原 Surfa 写入版本，不是官方 FreeSurfer 结果。与官方 FreeSurfer 的矩阵和辅助标签精度另见[同被试配准对照](../mni_aux_connected_20260927/README.md)：此前 LTA 矩阵最大差为 5.06×10⁻⁵ 体素，MCA/dura、静脉窦和 finalsurfs 体素相同。此次仅更换写入方式，不改变上游配准误差。

对应的官方命令链是 `mri_synthmorph -m affine` 后接两次 `mri_concatenate_lta`；完整命令和输入/输出见 [MNI 辅助链说明](../../../../docs/recon_all/MNI_AUX_CHAIN.md)。目前 SynthMorph 内部仍使用 Surfa；这项验证只验收 LTA 写入替换。
