"""Capture one complete real compact lookup, then intentionally stop the run."""

import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch


class Captured(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--stage", choices=("first", "intensity"), default="first")
    args, worker_args = parser.parse_known_args()
    if args.capture_directory.exists():
        parser.error("use a new private capture directory")
    args.capture_directory.mkdir(parents=True)
    import fnit.gems.rasterize as raster
    original = raster._compact_lookup
    in_intensity = False
    if args.stage == "intensity":
        import fnit.gems.core as core
        original_intensity = core.rasterize_priors_compact

        def intensity_raster(*function_args, **function_kwargs):
            nonlocal in_intensity
            previous = in_intensity
            in_intensity = True
            try:
                return original_intensity(*function_args, **function_kwargs)
            finally:
                in_intensity = previous

        core.rasterize_priors_compact = intensity_raster

    def capture(vertices, tetrahedra, valid_mask, block_index, current_geometry=None, tolerance=2e-5, **kwargs):
        result = original(vertices, tetrahedra, valid_mask, block_index, current_geometry, tolerance, **kwargs)
        if args.stage == "intensity" and not in_intensity:
            return result
        batches, _ = block_index.device_compact_batches(valid_mask, vertices.device, vertices.dtype)
        origins, inverses, selected, points, covered, reorder = result
        if current_geometry is not None:
            singular = current_geometry.singular
        else:
            cells = vertices[tetrahedra]
            edges = torch.stack([cells[:, index] - cells[:, 0] for index in (1, 2, 3)], -1)
            _, info = torch.linalg.inv_ex(edges, check_errors=False)
            singular = (info != 0) | (torch.linalg.det(edges).abs() <= 1e-10)
        np.savez_compressed(args.capture_directory / "geometry.private.npz",
                            origins=origins.detach().numpy(), inverses=inverses.detach().numpy(),
                            singular=singular.detach().numpy())
        for index, (point, ids, mask, _, rows) in enumerate(batches):
            np.savez_compressed(args.capture_directory / f"batch_{index:04d}.private.npz",
                                points=point.detach().numpy(), ids=ids.detach().numpy(),
                                mask=mask.detach().numpy(), rows=rows.detach().numpy())
        np.savez_compressed(args.capture_directory / "reference.private.npz",
                            selected=selected.detach().numpy(), points=points.detach().numpy(),
                            covered=covered.detach().numpy(), reorder=reorder.detach().numpy())
        (args.capture_directory / "capture.public.json").write_text(json.dumps({
            "diagnostic_only": True, "pipeline_completed": False,
            "capture_scope": "one complete compact lookup from unmodified real-input worker",
            "capture_stage": args.stage,
            "batch_count": len(batches), "real_point_count": selected.numel(),
            "tolerance": tolerance, "floating_dtype": str(vertices.dtype),
            "threads": torch.get_num_threads()}, indent=2) + "\n")
        raise Captured()

    raster._compact_lookup = capture
    sys.argv = [str(args.worker), *worker_args]
    try:
        runpy.run_path(str(args.worker), run_name="__main__")
    except Captured:
        print("complete real compact lookup saved privately; pipeline intentionally stopped")


if __name__ == "__main__":
    main()
