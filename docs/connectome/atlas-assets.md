# Connectome atlas 文件和许可

[返回 pipeline 用法](README.md) · [文件清单](../../src/fnit/connectome/atlas_manifest.json)

FNIT 固定 GitHub Release `assets-v1` 已核查，目前没有下表的 atlas。用户可按需下载仓库 `assets/connectome/` 中有明确再分发许可的文件；安装器逐个核对清单中固定的字节数和 SHA-256，不下载未选择的 Schaefer 分辨率。

```bash
ATLAS_TEMPLATES=/data/atlas/fnit  # 用户可写的模板目录
fnit-setup-connectome-atlases \
  --atlas schaefer200+tian-s1 schaefer500+tian-s4 \
  --output-dir "$ATLAS_TEMPLATES"
```

在 `fnit UKBConnectome_pipeline` 命令加 `--download-atlases` 可在首次使用时执行同一校验；省略 `--atlas-templates-dir` 时下载到本次 `--output-dir/atlas_templates`。`fs-aparc` 不需下载。已存在的同名文件若大小或 SHA-256 不符会报错，不会悄悄覆盖。

| 选择 | 下载的 atlas 文件 | 来源及条件 |
|---|---|---|
| `aparc+tian-s1`、`aparc.a2009s+tian-s1` | Tian S1 3T NIfTI 与名称表 | [Tian 原作者仓库](https://github.com/yetianmed/subcortex)的 [Melbourne Subcortex Atlas 许可](https://github.com/yetianmed/subcortex/blob/master/license.txt)：可再分发，发表时引用 Tian 等（2020） |
| `schaefer200+tian-s1` | 上述 Tian S1，加双半球 Schaefer200 `7Networks` 注释 | [CBIG Schaefer2018](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal)；[MIT 许可](https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md) |
| `schaefer500+tian-s4`、`schaefer1000+tian-s4` | Tian S4 与对应双半球 Schaefer 注释 | 同上 |
| `glasser+tian-s1/s4` | 目前不通过 FNIT 镜像分发 | 从 [HCP/BALSA 原作者资源](https://balsa.wustl.edu/)取得 Glasser dlabel 与球面；当前流程还需 Workbench `wb_command` |

`fsaverage_dir` 的 `surf/{lh,rh}.sphere.reg` 由 FreeSurfer fsaverage subject 提供；`mni_template` 必须与所选 Tian 标签的体素网格一致。它们不是上述 atlas 包的一部分，用户应从有权使用的原始软件或网站获取。SynthMorph 权重沿用 FNIT [固定 Release 与原作者清单](../../src/fnit/weights.py)的大小/SHA-256 校验。镜像的 Tian 压缩图像取自公开 [UKB-connectomics 模板目录](https://github.com/sina-mansour/UKB-connectomics/tree/main/data/templates/atlases)；解压后的 S1/S4 `.nii` 与 [Tian 原作者固定提交](https://github.com/yetianmed/subcortex/tree/dcad93421ea8021d6c5738df0a915a2223cd82aa/Group-Parcellation/3T/Subcortex-Only)逐字节一致，两个名称表也一致。六个 Schaefer `.annot` 均与 [CBIG 固定提交](https://github.com/ThomasYeoLab/CBIG/tree/35b5664bec8822e2f77da5e090e96f91d0095be6/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label)逐字节一致。全部镜像文件的尺寸和 SHA-256 见机器清单。不要把受试者 T1/DWI 或受控 UKB 数据放入 atlas 模板目录。
