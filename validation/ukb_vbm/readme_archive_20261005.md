# README历史归档（2026-10-05）

本页保留文档整理前完整正文，原始正文SHA-256：`e91d0278216ea905ce03f7302c08c2e4978007cd6e74822b619834632e07666f`。仅修正移位后的相对链接；最新用户手册见[功能README](../../docs/ukb_vbm/README.md)。历史测量仍绑定原源码与输入，不改标当前main。

# UK Biobank v1.5 VBM 参考

[返回主页](../../README.md) · [FastVBM 接口](../../docs/fast_vbm/README.md) · [当前验证](../fast_vbm/README.md)

本页只记录 FastVBM 当前实现所对应的 UK Biobank/FSL 方法和公开模板。旧的实验注册器、研究脚本和聚合报告已经删除。

## 官方 VBM 步骤

UK Biobank v1.5 的 `bb_struct_init` 从原始 T1 开始，执行视野裁剪、BET、标准空间掩膜回投和 FAST。随后 `bb_vbm` 对 GM PVE 运行：

```bash
fsl_reg T1_brain_pve_1.nii.gz template_GM.nii.gz   T1_GM_to_template_GM -fnirt   "--config=GM_2_MNI152GM_2mm.cnf --jout=T1_GM_JAC_nl"
fslmaths T1_GM_to_template_GM -mul T1_GM_JAC_nl   T1_GM_to_template_GM_mod -odt float
```

FastVBM 以相同的输出角色结束：GM、warped GM、nonlinear-only Jacobian 和 modulated GM。当前两条分支共用 SynthStrip/TorchFAST、TorchFLIRT、FSL 坐标转换、TorchApplyWarp、Jacobian 与 modulation，只在非线性估计器上选择 PyTorch SynthMorph 或 TorchFNIRT。逐阶段对应和输入输出见 [FastVBM 文档](../../docs/fast_vbm/README.md)。

源码核对基于 UKB v1.5 commit `0e39a7f7eb76b55437942bfa3073512506b6c8fa`；`bb_structural_pipeline/bb_vbm` 的 SHA-256 为 `efdca88961dad9eeec52e15e2d26ad5f807b2c0bd3b41c547990ba78ebb7753f`。当前 FSL 参考环境为 6.0.7.4，因此验证范围是 v1.5 方法与参数复现，不包含逐字节等价声明。

## 公开模板

UKB ancillary archive：

```text
https://www.fmrib.ox.ac.uk/ukbiobank/fbp/templates/dckr_build/DATA_public.tar.gz
SHA-256: 52c2349270d4d19b8de6a0d136270e74f6a68379d02bda6635a92306c18e2319
size: 689432077 bytes
```

FastVBM 需要其中的 `templates/template_GM.nii.gz`；TorchFNIRT 参考运行还使用模板网格的脑掩膜。仓库与 wheel 不分发该 archive 或模板。

```bash
curl -L -o DATA_public.tar.gz   https://www.fmrib.ox.ac.uk/ukbiobank/fbp/templates/dckr_build/DATA_public.tar.gz
echo '52c2349270d4d19b8de6a0d136270e74f6a68379d02bda6635a92306c18e2319  DATA_public.tar.gz'   | sha256sum -c -
mkdir -p assets
tar -xzf DATA_public.tar.gz -C assets --strip-components=1   templates/template_GM.nii.gz   templates/MNI152_T1_1mm_brain_mask.nii.gz   templates/MNI152_T1_1mm_brain_mask_dil_GD7.nii.gz
```

公开 `template_GM.nii.gz` 的 SHA-256 为 `ab933db7455d7c4b88624d54f41a3065be4ba4289d00b9230daec0cdb1597a77`。模型权重由 [权重配置](../../docs/WEIGHTS.md)单独下载；模板不是模型权重。

## 当前证据边界

旧 FastVBM 端到端报告不再对应当前源码，已从仓库移除。当前 TorchFNIRT 与 SynthMorph 两个后端已用同一例真实 T1w 完成 raw-to-modulated-GM 复跑；两条路径的 shape、affine 和 dtype 合同通过，但连续值数值等价均未通过，因此不能把计时写成等价实现的加速比。精度、阶段时间、显存、源码哈希和两张当前图见 [FastVBM 验证页](../fast_vbm/README.md)；组件级 TorchFNIRT/TBSS 诊断见 [dMRI 验证页](../dmri_pipeline/README.md)。

## 来源

- [UK Biobank pipeline v1](https://git.fmrib.ox.ac.uk/open-science/analysis/UK_biobank_pipeline_v_1)
- [FMRIB UK Biobank pipeline and ancillary files](https://www.fmrib.ox.ac.uk/ukbiobank/fbp/)
- [FNIRT user guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/fnirt/user_guide.html)
- [FSL FNIRT source](https://git.fmrib.ox.ac.uk/fsl/fnirt)
