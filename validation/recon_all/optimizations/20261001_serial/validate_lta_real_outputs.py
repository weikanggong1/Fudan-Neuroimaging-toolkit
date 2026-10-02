"""验证两例已完成FNIT整例的六个LTA读取，不计算MRI、不运行官方软件。
config为比较配置JSON，使用cases中的id/candidate；output为新JSON，commit为实际源码。
输入仅冻结自产LTA，输出SHA、坐标类型、矩阵差异、source/target网格shape及耗时秒。
world平移mm、线性系数无量纲；voxel矩阵以体素为坐标。读取后不转换矩阵。
缺文件、输出已存在、格式无效、非有限几何或矩阵改变时抛异常。
原格式对应FreeSurfer LTA；这是内部格式步骤，无独立原软件CLI。真实结果不代表新整例。
"""
import argparse,hashlib,inspect,json,platform,time
from pathlib import Path
import numpy as np
from fnit._transforms import load_lta

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--commit",required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    config=json.loads(a.config.read_text())
    tick=time.perf_counter();rows=[]
    for case in config["cases"]:
        for name in ["transforms/synthmorph.mni305/aff.lta",
                     "transforms/synthmorph.1.0mm.1.0mm/aff.lta",
                     "transforms/synthmorph.1.0mm.1.0mm/reg.targ_to_invol.lta"]:
            path=Path(case["candidate"])/"mri"/name
            data=path.read_bytes();lines=data.decode().splitlines();start=lines.index("1 4 4")+1
            matrix=np.asarray([[float(v) for v in row.split()] for row in lines[start:start+4]],np.float64)
            affine=load_lta(path)
            if not np.array_equal(matrix,affine.matrix):raise ValueError("matrix changed")
            if not np.isfinite(affine.source.affine).all() or not np.isfinite(affine.target.affine).all():
                raise ValueError("nonfinite geometry")
            rows.append({"case":case["id"],"path":str(path),"input_sha256":hashlib.sha256(data).hexdigest(),
                         "space":affine.space,"matrix_different_elements":0,
                         "source_shape":affine.source.shape,"target_shape":affine.target.shape})
    result={"code_commit":a.commit,"host":platform.node(),"scope":"two real FNIT subjects, six LTA file-reader regression; no new whole execution",
            "seconds_including_read":time.perf_counter()-tick,
            "source_sha256":hashlib.sha256(Path(inspect.getfile(load_lta)).read_bytes()).hexdigest(),
            "benchmark_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_sha256":hashlib.sha256(a.config.read_bytes()).hexdigest(),"cases":rows,"passed":True}
    a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result))
if __name__=="__main__":main()
