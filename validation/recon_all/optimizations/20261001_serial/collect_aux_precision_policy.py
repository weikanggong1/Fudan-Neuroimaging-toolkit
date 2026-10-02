"""汇总两例冻结自产输入的实际GPU精度回归；不修改生产文件。
--root为执行目录，--commit为真实源码版本，--output必须不存在。
输出实际前向、模型/影像哈希、CPU/GPU标签Dice和阶段/进程计时。
TF32默认对照必须与修复前GPU逐体素一致；FP32对照须验证实际前向开关。
"""
import argparse, hashlib, json
from pathlib import Path
import nibabel as nib
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compare(before, after):
    x, y = nib.load(before), nib.load(after)
    u, v = np.asarray(x.dataobj), np.asarray(y.dataobj)
    if u.shape != v.shape:
        raise ValueError("label grids have different shapes")
    labels = np.union1d(u, v)
    return {
        "different_voxels": int(np.count_nonzero(u != v)),
        "geometry_equal": bool(np.array_equal(x.affine, y.affine)),
        "dtype_equal": bool(x.get_data_dtype() == y.get_data_dtype()),
        "before_sha256": sha(before), "after_sha256": sha(after),
        "label_dice": {str(int(i)): float(2*np.count_nonzero((u == i)&(v == i)) /
            (np.count_nonzero(u == i)+np.count_nonzero(v == i))) for i in labels if i != 0},
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    report = {"scope": "frozen_self_generated_input; cache_enabled; not_whole_speedup",
              "code_commit": a.commit, "collector_sha256": sha(__file__),
              "production_default": "TF32 unchanged", "cases": {}}
    for sub in ["01", "02"]:
        policies = {}
        for policy, cudnn_tf32, matmul_tf32 in [("1", True, True), ("0", False, False), ("convfp32", False, True)]:
            tag = "aux_policy_c757_sub"+sub+"_"+str(policy)
            run = json.loads((a.root/(tag+".json")).read_text())
            monitor = json.loads((a.root/(tag+"_monitor")/"monitor.json").read_text())
            assert run["code_commit"] == a.commit
            forwards = run["precision"]["entowm"] + run["precision"]["mni_aux"]["auxiliary_forwards"]
            assert len(forwards) == 5, "missing actual forward records"
            for item in forwards:
                assert item["device"] == "cuda:0"
                assert item["input_dtype"] == "torch.float32"
                assert item["model_dtypes"] == ["torch.float32"]
                assert not item["autocast"]["enabled"]
                assert item["matmul_tf32"] == matmul_tf32
                if item["model"] != "affine":
                    assert item["cudnn_tf32"] == cudnn_tf32, "cuDNN policy overridden"
            comparisons = {}
            for reference in ["oldcpu", "newgpu"]:
                comparisons[reference] = {}
                for name in ["entowm", "mca-dura", "vsinus"]:
                    row = compare(a.root/("stage1r2_sub"+sub+"_"+reference)/"mri"/(name+".mgz"),
                                  a.root/tag/"mri"/(name+".mgz"))
                    assert row["geometry_equal"] and row["dtype_equal"]
                    if policy == "1" and reference == "newgpu":
                        assert row["different_voxels"] == 0, "default GPU regression"
                    comparisons[reference][name] = row
            policies[str(policy)] = {"requested_cudnn_tf32": cudnn_tf32,
                "requested_matmul_tf32": matmul_tf32, "actual_forward_policy_valid": True,
                "run": run, "monitor": monitor, "comparisons": comparisons}
        assert policies["0"]["run"]["input_sha256"] == policies["1"]["run"]["input_sha256"]
        assert policies["0"]["run"]["weight_sha256"] == policies["1"]["run"]["weight_sha256"]
        assert policies["convfp32"]["run"]["input_sha256"] == policies["1"]["run"]["input_sha256"]
        report["cases"][sub] = policies
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({sub: {policy: {"seconds": item["run"]["seconds"],
        "cpu_differences": {n: row["different_voxels"] for n,row in item["comparisons"]["oldcpu"].items()}}
        for policy,item in policies.items()} for sub,policies in report["cases"].items()}), flush=True)

if __name__ == "__main__": main()
