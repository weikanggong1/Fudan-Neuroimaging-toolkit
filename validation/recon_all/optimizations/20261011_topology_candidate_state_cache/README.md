# 完整候选状态缓存：真实冻结阶段

本轮性能目标为减少候选全网格状态装配。默认`state_context=None`保持旧路径；新显式上下文完整验证当前全部有效行、CSR、面前缀/append-only，动态属性和容量尾部保持。生产及live GA未切换。

v2完整ABBA严格162字段/控制全相同，但CLI慢5.6761%；v3全域ndarray/list打包分流后热装配约0.42–0.45秒、热B完整cycle中位观察缩短8.599%，完整CLI仍慢4.3799%、API慢3.5864%。二者均不能称完整阶段提速；保留所有失败性能。v1启动器只上传未执行，因未声明环境变量改新v2。四child每版EXIT0。

本轮仅公开ds000114 sub07 ses-test LH/FNIT79上游的冻结候选与末段序列，完整110534V/216860F/38inside/99selected；不是新125k/67inside病例、全GA或rawT1整例。源首次候选/完整末态自身重复门保持，最终网格QC与整体指标等效未评估。参考after仅在输出保存后读取。

- `reports/v2/report.json`、`reports/v3/report.json`：固定字段白名单导出，完整比对/原私有报告SHA/测试源码与输入SHA。私有影像、命令/日志、原始状态NPZ及配置留在授权环境。
- `summary.json`：上述报告的确定性派生数值；非性能门更改。
- `contracts.xml/json`：17tests+9subtests10.52秒，本地线程/别名/失败契约，不是MRIbenchmark。
- `environment_observation.json`：同主机/同四核四线程、GPU隐藏，RSS未正式采样，显存记null。
- `export_public.py`：固定白名单导出脚本，只导出必要数值与哈希。

源码实测绑定v3 bundle manifest5674352e…，state64d57c499…、cycle4ffe4373…；publication commit不改标运行祖先/overlay。完整C++程序不在生产，未新增依赖，未更改精度/轮数/质量门/输出。
