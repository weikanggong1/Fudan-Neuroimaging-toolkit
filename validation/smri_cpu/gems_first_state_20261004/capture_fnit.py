"""Capture exactly the first real recipe mesh objective; never finish a fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import nibabel as nib
import numpy as np
import torch

from fnit.gems import core, optim, rasterize
from fnit.gems.context import SubregionContext
from fnit.gems.recipes.thalamus import ThalamusRecipe
from fnit.gems.recipes.hippo_amygdala import HippoAmygdalaRecipe


class Captured(Exception):
    pass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("t1", "aseg", "wmparc", "atlas", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--structure",choices=("thalamus","hippo-amygdala-left","hippo-amygdala-right"),required=True)
    args = parser.parse_args()
    expected = {"norm":"001021a47f102bb9772f7f736b65f74740334e86dfd1afd0386ca250cb25b698",
                "aseg":"ea02b3cd278c8eb229a4cd6a5bec982a0a6175f7d34ab3f651696902e1a259f8",
                "wmparc":"bd45c6371c7d389cfc5d9ce5cabe321e362d32cce6feb35fdaea81e7606a469c"}
    for name,path in (("norm",args.t1),("aseg",args.aseg),("wmparc",args.wmparc)):
        if digest(path) != expected[name]:
            raise RuntimeError("fixed real input hash mismatch: "+name)
    args.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(8)
    import numba
    numba.set_num_threads(min(8,numba.config.NUMBA_NUM_THREADS))
    context = SubregionContext.prepare(args.t1,need_coarse=True,need_parc=False,
                                      coarse_segmentation=args.aseg,wmparc=args.wmparc,device="cpu")
    recipe = (ThalamusRecipe("thalamus",args.atlas) if args.structure=="thalamus"
              else HippoAmygdalaRecipe(args.structure.rsplit("-",1)[1],args.atlas))
    recipe.set_optimization_profile("fast")
    original_call = core.TorchGEMS.__call__
    captured = {}

    def capture_call(model, image, **options):
        captured.update(model=model,image=image,options=options)
        raise Captured()

    core.TorchGEMS.__call__ = capture_call
    try:
        recipe.run(context,torch.device("cpu"))
    except Captured:
        pass
    finally:
        core.TorchGEMS.__call__ = original_call
    model,image,options = (captured[name] for name in ("model","image","options"))
    atlas = model.atlas
    fixed = options["fixed_gaussians"]
    classes = np.asarray(options["label_classes"])
    alphas = np.asarray(options["fit_alpha_stages"][0][0])
    np.savez_compressed(args.output/"shared_input.npz", image=image.numpy(),
        vertices=atlas.vertices.astype(np.float32),reference=atlas.reference_vertices.astype(np.float32),
        tetrahedra=atlas.tetrahedra,alphas=alphas,can_move=atlas.can_move,
        means=fixed.means.numpy().astype(np.float64),variances=fixed.covariances.numpy().astype(np.float64),
        stiffness=atlas.stiffness,boundary_transform=options["boundary_transform"],
        original_reference=recipe._reference_vertices,classes=classes,
        raw_alphas=atlas.alphas,label_ids=atlas.label_ids)
    original_raster = core.rasterize_priors_compact

    def capture_raster(*values,**kwargs):
        priors,coverage = original_raster(*values,**kwargs)
        # The final call before the first closure returns is the mesh objective,
        # with exactly its selected alpha stage, mask and current geometry.
        np.save(args.output/"fnit_priors.npy",priors.detach().numpy())
        np.save(args.output/"fnit_coverage.npy",coverage.detach().numpy())
        return priors,coverage

    def first_step(optimizer,closure,*,cache_key=None):
        cost = closure()
        vertices = optimizer.param_groups[0]["params"][0]
        np.save(args.output/"fnit_gradient.npy",vertices.grad.detach().numpy())
        np.save(args.output/"fnit_vertices.npy",vertices.detach().numpy())
        captured["first_cost"] = float(cost)
        raise Captured()

    original_step = optim.CachedArmijoLBFGS.step
    core.rasterize_priors_compact = capture_raster
    optim.CachedArmijoLBFGS.step = first_step
    try:
        original_call(model,image,**options)
    except Captured:
        pass
    finally:
        core.rasterize_priors_compact = original_raster
        optim.CachedArmijoLBFGS.step = original_step
    if "first_cost" not in captured:
        raise RuntimeError("did not reach the first real mesh objective")
    report = {"status":"first_objective_captured", "structure":args.structure,
              "scope":"real same-input synthetic-label recipe, first objective only; no optimizer update or complete fit",
              "host":socket.gethostname(),"affinity":sorted(os.sched_getaffinity(0)),
              "torch_threads":torch.get_num_threads(),"numba_threads":numba.get_num_threads(),
              "first_cost":captured["first_cost"],"image_shape":list(image.shape),
              "valid_voxels":int((image!=0).sum()),"vertices":len(atlas.vertices),
              "tetrahedra":len(atlas.tetrahedra),"classes":int(alphas.shape[1]),
              "background_class":int(classes[options["background_channel"]]),
              "source":{str(path.relative_to(Path(core.__file__).parent)):digest(path)
                        for path in sorted(Path(core.__file__).parent.rglob("*.py"))},
              "inputs":{name:digest(path) for name,path in (("norm",args.t1),("aseg",args.aseg),
                                                          ("wmparc",args.wmparc),("mesh",args.atlas/"AtlasMesh.gz"),
                                                          ("lut",args.atlas/"compressionLookupTable.txt"),
                                                          ("atlas_dump",args.atlas/"AtlasDump.mgz"))},
              "outputs":{p.name:digest(p) for p in args.output.iterdir() if p.is_file()}}
    (args.output/"report.public.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({name:report[name] for name in ("status","structure","first_cost","valid_voxels")}))


if __name__ == "__main__":
    main()
