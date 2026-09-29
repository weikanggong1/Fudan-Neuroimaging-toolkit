# 空间转换真实数据验证（2026-09-29）

## 数据和环境

- MNI 输入：FSL 发布的 `MNI152_T1_1mm.nii.gz`，SHA-256 `d1f03e160c2548592a01d98d44d7e0ffa8a8ef58bc9b07d26369e214ba170edb`。此文件只用于对照测试；FNIT 不调用 FSL 命令。
- 表面输入：fsaverage6 平均沟深图，左侧 SHA-256 `fa7e34c73a13a714eb1caf9c751b0d0fd4242bf2ba81a70293d700b8bb107ae5`，右侧 `dbdc00ca7c8861ce792d2636f7348fde22fae9a5f34623a7162e305823f8acd7`。
- 空间资产：HCPpipelines `f8cac6892f88bdf889d644711ff038198eb81533` 的 2017 年球面和面积文件；CBIG `v0.18.1-Update_stable_project_unit_test` 的 FS5.3 RF-ANTs 映射。全新目录在线安装 38 个文件，38 个 SHA-256 均通过；随后从该目录完成了 MNI→fsLR32k 运行。
- 本地 CPU：Intel Xeon Gold 6254，PyTorch 2.5.1、Nibabel 5.4.2、SciPy 1.17.1、Workbench 1.5.0、MATLAB R2023b。计时采用 `time.perf_counter()` 或 MATLAB `tic/toc`；官方 MATLAB 函数计时不含 MATLAB 启动。

## 与原实现对照

| 操作 | FNIT CPU | 官方 | 差异 |
|---|---:|---:|---|
| MNI152 T1 1 mm → fsaverage164k，线性 | 0.4125 s | `CBIG_RF_projectMNI2fsaverage` 2.553 s | 左 MAE 0.001314、最大 0.021764；右 MAE 0.001089、最大 0.020666；两侧相关系数均 >0.99999999999。 |
| fsaverage164k → MNI152 1 mm，最近顶点 | 4.1071 s | `CBIG_RF_projectfsaverage2Vol_single` 20.732 s | 体积仿射相同，1,029,656 个非零皮层体素逐体素相同；整图最大绝对差 0。 |
| fsaverage6 平均沟深 → fsLR32k，面积校正 | 0.7721 s | 独立运行 Workbench `-metric-resample ADAP_BARY_AREA` 0.7733 s | 双半球逐顶点相同；最大绝对差 0。FNIT 在这一步调用相同的 Workbench 命令，因此计时主要是命令启动与读写。 |

CPU 正向输出为 float32，CBIG MATLAB 输出为 double；上表的非零差值反映这一步精度及插值运算顺序。反向测试把 CBIG 正向生成的真实 T1 表面值同时交给两套程序，单独测量反向映射差异。CBIG 反向输出与 FNIT 的 1 mm NIfTI 逐体素相同。

## 分辨率和帧检查

| 路径 | 结果 |
|---|---|
| MNI152 1 mm → fsaverage41k、fsLR32k | 双侧分别输出 40,962、32,492 个顶点。 |
| 原始 MNI T1 经线性重采样形成的 0.5 mm 图 → fsaverage3k | 双侧各 2,562 个顶点。0.5 mm 图不是独立扫描。 |
| fsLR32k → fsLR59k、fsaverage10k | 双侧分别输出 59,292、10,242 个顶点。 |
| fsLR32k → MNI152 | CBIG 1 mm 掩膜内 1,029,656 个非零体素。 |
| fsaverage164k → MNI152 2 mm、0.5 mm 参考网格 | 分别得到 `(91,109,91)`、`(363,435,363)`，皮层非零体素数分别为 128,715、8,255,692；耗时 0.97、15.62 秒。 |
| MNI T1 两帧图 → fsLR32k → MNI152 两帧图 | GIFTI 每侧两帧、NIfTI `(256,256,256,2)`；第二帧为第一帧的 0.5 倍，两端最大比例误差为 0。 |
| fsaverage6 二值皮层标签 → fsLR32k | 双侧标签 GIFTI 可由 Workbench `-label-resample` 读取。 |

## 复现方式

先运行 `fnit-setup-space-assets --output-dir /tmp/fnit-space-assets`，再按[功能说明](../../docs/space_conversion/README.md)调用 `convert_space`。官方数值对照使用以下两个 CBIG 函数，均指定上述发布的 RF-ANTs 文件：

```matlab
[lh, rh] = CBIG_RF_projectMNI2fsaverage('/path/MNI152_T1_1mm.nii.gz', 'linear', lh_map, rh_map);
[projected, projected_seg] = CBIG_RF_projectfsaverage2Vol_single(lh, rh, 'nearest', reverse_map, cortex_mask);
```

Workbench 对照命令见功能说明。所有计时都在单被试、双半球运行；运行前后的路径、输出顶点数、仿射和数值均已直接读取核对。大型原始模板与生成影像未加入 Git 仓库。
