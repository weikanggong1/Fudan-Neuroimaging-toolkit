"""Explicit full native registration using FNIT's validated search hotspot build."""
from pathlib import Path
import hashlib
import os
import subprocess


def run_cached_em_register(binary: str | Path, mri: str | Path,
                            atlas: str | Path, assets: str | Path, *,
                            binary_sha256: str) -> Path:
    """完整EM入口；仅搜索似然采用缓存CPU热点，梯度/优化/终止均为原生实现。

    binary为独立Conda固定源码构建产物，binary_sha256为已验证manifest中的
    64位SHA-256。mri含自产nu.mgz、brainmask.mgz和transforms目录；atlas为
    已校验单通道T1 GCA，assets为授权模板目录。输出transforms/talairach.lta
    为source voxel到atlas voxel的4×4变换，LTA携带两侧网格几何。
    FS_LICENSE从已授权进程环境继承；不复制或保存许可证。
    输入缺失、程序哈希不符、非正4线程预算或程序失败明确抛异常。
    总线程由调用者设置OMP_NUM_THREADS=4；没有GPU或低精度选项。
    """
    binary=Path(binary).resolve();mri=Path(mri).resolve()
    if not binary.is_file() or not os.access(binary,os.X_OK):
        raise FileNotFoundError(binary)
    if len(binary_sha256)!=64 or hashlib.sha256(binary.read_bytes()).hexdigest()!=binary_sha256:
        raise ValueError('unverified native cached executable SHA-256')
    capabilities=subprocess.run([str(binary)],env=dict(os.environ,FNIT_GCA_QUERY_CAPABILITIES='1'),capture_output=True,text=True,check=True)
    import json
    if not json.loads(capabilities.stdout).get('fnit_gca_cached_search'):
        raise ValueError('native cached search capability missing')
    for path in (mri/'nu.mgz',mri/'brainmask.mgz',Path(atlas)):
        if not path.is_file():raise FileNotFoundError(path)
    if os.environ.get('OMP_NUM_THREADS')!='4':
        raise ValueError('declare OMP_NUM_THREADS=4 before the full cached EM call')
    (mri/'transforms').mkdir(exist_ok=True)
    env=dict(os.environ,FREESURFER_HOME=str(Path(assets).resolve()),FNIT_GCA_SCORER='cpu_cached')
    subprocess.run([str(binary),'-uns','3','-mask','brainmask.mgz','nu.mgz',str(Path(atlas).resolve()),'transforms/talairach.lta'],cwd=mri,env=env,check=True)
    output=mri/'transforms/talairach.lta'
    if not output.is_file():raise FileNotFoundError(output)
    return output
