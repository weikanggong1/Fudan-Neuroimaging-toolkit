"""Isolated official ABI oracle on exactly the captured real GEMS inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import sysconfig

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared",type=Path,required=True)
    parser.add_argument("--mesh",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    # Official extension is CPython3.8. Keep its loader out of FNIT production.
    sys.path.insert(0,"/public/software/apps/Freesurfer/8.2.0-1/python/lib/python3.8/site-packages/samseg/gems")
    import gemsbindings as gems
    data = np.load(args.shared)
    points = np.asfortranarray(data["vertices"].astype(np.float64))
    reference = np.asfortranarray(data["reference"].astype(np.float64))
    alphas = np.asfortranarray(data["alphas"].astype(np.float32))
    image_array = data["image"]
    valid = image_array != 0
    image = gems.KvlImage(np.asfortranarray(image_array))
    roundtrip = np.asarray(image.getImageBuffer())
    if not np.array_equal(roundtrip,image_array):
        raise RuntimeError("official image buffer changed grid or values")
    initial_collection = gems.KvlMeshCollection()
    initial_collection.read(str(args.mesh))
    raw_reference = np.asarray(initial_collection.reference_mesh.points)
    raw_can_move = np.asarray(initial_collection.reference_mesh.can_moves)
    raw_reference_exact = np.array_equal(raw_reference,data["original_reference"])
    raw_reference_float32_exact = np.array_equal(raw_reference,data["original_reference"].astype(np.float32).astype(np.float64))
    if not (raw_reference_exact or raw_reference_float32_exact):
        raise RuntimeError("raw atlas node order/reference points differ")
    if not np.array_equal(raw_can_move,data["can_move"]):
        raise RuntimeError("raw atlas node flags differ")
    transform = np.eye(4)
    # Production converts this input to FP32 before its QR projection. Keep
    # the oracle's supplied matrix identical, independently of QR arithmetic.
    transform[:3,:3] = data["boundary_transform"].astype(np.float32).astype(np.float64)
    transform = gems.KvlTransform(np.asfortranarray(transform))
    calculator = gems.KvlCostAndGradientCalculator(
        typeName="AtlasMeshToIntensityImage",images=[image],boundaryCondition="Sliding",
        transform=transform,means=np.asfortranarray(data["means"]),
        variances=np.asfortranarray(data["variances"]),
        mixtureWeights=np.ones(len(data["means"]),dtype=np.float32),
        numberOfGaussiansPerClass=np.ones(len(data["means"]),dtype=np.int32))
    results = {}
    for name,stiffness in (("total",float(data["stiffness"])),("data",0.)):
        collection = gems.KvlMeshCollection()
        collection.read(str(args.mesh))
        collection.set_positions(reference,[points])
        collection.k = stiffness
        mesh = collection.get_mesh(0)
        mesh.alphas = alphas
        if not np.array_equal(mesh.alphas,alphas):
            raise RuntimeError("official alpha setter changed shared values")
        if not np.array_equal(mesh.points,points):
            raise RuntimeError("mesh setter changed the shared points")
        cost,gradient = calculator.evaluate_mesh_position(mesh)
        np.save(args.output/("native_"+name+"_gradient.npy"),gradient)
        results[name] = float(cost)
        if name=="data":
            priors = np.asarray(mesh.rasterize_values(list(image_array.shape),np.asfortranarray(alphas.astype(np.float64))))
            rounded = np.asarray(mesh.rasterize(list(image_array.shape)))
            np.save(args.output/"native_priors.npy",priors[valid].T)
            np.save(args.output/"native_priors_uint16.npy",rounded[valid].T)
            results["continuous_dtype"] = str(priors.dtype)
            results["continuous_shape"] = list(priors.shape)
            results["uint16_dtype"] = str(rounded.dtype)
    report = {"status":"first_objective_evaluated","scope":"isolated official cost/prior oracle; no fit",
              "host":socket.gethostname(),"affinity":sorted(os.sched_getaffinity(0)),
              "python":sys.version,"python_executable":sys.executable,
              "SOABI":sysconfig.get_config_var("SOABI"),"numpy":np.__version__,
              "native_binding":{"path":gems.__file__,"sha256":digest(Path(gems.__file__))},
              "shared_input_sha256":digest(args.shared),"mesh_sha256":digest(args.mesh),
              "raw_reference_original_exact":bool(raw_reference_exact),
              "raw_reference_after_float32_exact":bool(raw_reference_float32_exact),
              "node_flags_exact":True,"image_roundtrip_exact":True,
              "common_points":"captured FNIT FP32 coordinates promoted to the official FP64 API; setter/getter exact",
              "common_reference":"captured FNIT FP32 reference promoted to the official FP64 API",
              "common_alphas":"captured first smoothed reduced FP32 alpha; setter/getter exact",
              "common_gaussians":"actual fixed Gaussian FP32 means/covariances promoted to official FP64 API",
              "common_boundary":"actual FNIT FP32 boundary matrix promoted to official FP64 API",
              "mask":"same finite, nonzero FP32 image; native image buffer roundtrip exact",
              "cost":results,"outputs":{p.name:digest(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output/"report.public.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"cost":results}))


if __name__ == "__main__":
    main()
