"""Replay the saved v11 thalamus candidate-ID comparison without changing frozen sources."""
from pathlib import Path
import importlib.util
import hashlib
import json
import sys
import time
import nibabel as nib
import numpy as np
import torch

ROOT = Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_unified_20260930")
WORK = ROOT / "lookup_math_diag_v11_20261001"
FROZEN = ROOT / "source_v11"
PREPARED = Path("/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/fnit_subregions_plus_20260928/samseg_native_build_20260929/port_thalamus_debug")
sys.path.insert(0, str(FROZEN / "src"))
from fnit.gems.atlas import GEMSAtlas
from fnit.gems.rasterize import build_block_index

torch.set_num_threads(4)
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(.14, 0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device("cuda:0")
frozen_path = FROZEN / "src/fnit/gems/_raster_triton.py"
frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
assert frozen_hash == "b67c4b01a1c9effe11dd500284f51769330e0447d05feec09ffeaae8ff2abff7"
names = ("no_fusion_v11", "fusion_enabled", "explicit_fma_forward", "explicit_fma_reverse")
modules = {}
stats = {}
for name in names:
    path = WORK / (name + ".py")
    spec = importlib.util.spec_from_file_location("lookup_diag_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    modules[name] = module
    stats[name] = {"id_difference": 0, "covered_id_difference": 0, "coverage_difference": 0,
                   "examples": [], "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
image = nib.load(PREPARED / "processedImageMasked.mgz")
aligned = nib.load(PREPARED / "alignedAtlasImage.mgz")
atlas = GEMSAtlas.from_freesurfer(PREPARED / "warpedOriginalMesh.txt.gz",
                                ROOT / "atlases/thalamus/compressionLookupTable.txt")
atlas = atlas.transformed(np.linalg.inv(image.affine) @ aligned.affine, transform_reference=True)
data = torch.as_tensor(np.asarray(image.dataobj, np.float32).squeeze(), device=device)
vertices = torch.as_tensor(atlas.vertices, device=device, dtype=torch.float32)
tetrahedra = torch.as_tensor(atlas.tetrahedra, device=device)
cells = vertices[tetrahedra]
origins = cells[:, 0]
matrix = torch.stack((cells[:, 1]-origins, cells[:, 2]-origins, cells[:, 3]-origins), dim=-1)
inverse, info = torch.linalg.inv_ex(matrix, check_errors=False)
singular = (info != 0) | (torch.linalg.det(matrix).abs() <= 1e-10)
index = build_block_index(atlas.vertices, atlas.tetrahedra, tuple(data.shape), margin=3)
batches, _ = index.device_compact_batches(data > 0, device, torch.float32)
started = time.monotonic()
for number, (points, ids, candidate_mask, batch_ids, point_rows) in enumerate(batches):
    with torch.no_grad():
        rel = points[:, :, None, :] - origins[ids][:, None]
        w123 = torch.einsum("bcij,bpcj->bpci", inverse[ids], rel)
        weights = torch.cat((1.0-w123.sum(-1, keepdim=True), w123), dim=-1)
        score = weights.amin(-1).masked_fill((singular[ids] | ~candidate_mask)[:, None], -torch.inf)
        best_score, best = score.max(dim=2)
        expected_ids = ids[batch_ids, best].flatten()[point_rows]
        expected_coverage = (best_score >= -2e-5).flatten()[point_rows]
    for name, module in modules.items():
        got_ids, got_coverage = module.lookup_candidates(points, ids, candidate_mask, origins,
                                                        inverse, singular, point_rows, tolerance=2e-5)
        changed = got_ids != expected_ids
        covered_changes = changed & got_coverage & expected_coverage
        entry = stats[name]
        entry["id_difference"] += int(changed.sum())
        entry["covered_id_difference"] += int(covered_changes.sum())
        entry["coverage_difference"] += int((got_coverage != expected_coverage).sum())
        if covered_changes.any() and len(entry["examples"]) < 4:
            positions = torch.nonzero(covered_changes).flatten()[:4-len(entry["examples"])]
            for position in positions.cpu().tolist():
                a, b = int(expected_ids[position]), int(got_ids[position])
                entry["examples"].append({"batch": number,
                    "point": points.reshape(-1, 3)[point_rows[position]].cpu().tolist(),
                    "torch_id": a, "variant_id": b, "torch_vertices": atlas.tetrahedra[a].tolist(),
                    "variant_vertices": atlas.tetrahedra[b].tolist()})
result = {"frozen_source_sha256": frozen_hash, "torch": torch.__version__, "cuda": torch.version.cuda,
          "tf32": False, "shape": list(data.shape), "batches": len(batches),
          "real_points": sum(len(batch[4]) for batch in batches), "variants": stats,
          "diagnostic_wall_seconds": time.monotonic()-started,
          "peak_gpu_gib": torch.cuda.max_memory_allocated()/2**30,
          "frozen_source_unchanged": hashlib.sha256(frozen_path.read_bytes()).hexdigest() == frozen_hash}
# A replay preserves the original observed report.
(WORK / "candidate_id_comparison_replay.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result), flush=True)
