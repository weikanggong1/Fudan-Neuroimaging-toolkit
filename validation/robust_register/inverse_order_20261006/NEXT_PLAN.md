# 下一项有限实图计划（尚未派发）

目标：只测试源码顺序inverse替换，能否改善旧B的三项失败，不把18项矩阵合同视为影像验收。

1. 建独立冻结命名空间和进程。以原六个B文件为基线，五个文件字节不变；唯一数学delta是 `_sampling.native_inverse` 的局部委托及增加当前 `inverse_candidate.py`。不替换已安装或生产 `fnit`模块，不改成熟ca inverse，不读取旧native affine/fine points来初始化优化。
2. 复用原targetMask与reflectedAtlasDump的两个固定输入、shape/header/原SHA及原官方刚体/仿射输出。两stage分别完整读取自己的输入，刚体MGH保存并实际重读后运行仿射；没有新的官方命令、GEMS、recon-all或GPU。
3. 原sat50、Float图像/A/QR、Double小矩阵状态、每层5步、minimum16/maximum-1、stop0.01及原高分辨率策略不变；预期两stage×两层×5步，最多20次参数更新。所有实际level/step/停止原因必须报告。
4. 每个stage API独立有限进程≤60s，总科学范围≤300s；CPU8、原物理核、共用CPU锁、AS20GB、隐藏CUDA、清理注入环境。source/assets/flags开始结束固定。任何失败保原日志、exception、RC及partial，第一次数值gate失败停，不放宽门、不自动重复。
5. 保存各API与MGH/LTA I/O时钟/RSS、输出SHA；后验big-endian scorer明确Float32。仍按原20项正式门：源数据shape/dtype不变、LTA/source targetshape、内部header≤1e-5mm，原133点RMS≤0.001mm、max≤0.01mm、shared warp relL2≤1e-5、support差0。固定目标mask overlap/13字段另列；不是最终ROI验收。
6. worker与PLAN/freeze的源和输入/官方reference SHA准备后由root复核、协调一次派发。计划本身不执行数值，当前科学计数仍是6个矩阵/18项门，0新注册/0新采样。

未完成：实际rigid/affine/scorer worker、冻结source/input/reference binding及派发。不要从本文的资源预算声称已有新运行结果。
