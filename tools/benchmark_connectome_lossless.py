"""Compare exact connectome dataflow changes on real images and a fixed TCK.

The baseline is an isolated FNIT source checkout, never an external runtime.
Timings exclude image loading and equality checks. No synthetic inputs are
created. The saved report contains hashes and aggregates, not image data.
"""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import statistics
import time

import nibabel as nib
import numpy as np
import torch

from connectome_benchmark_common import _load, _sha256, _sync
import fnit.connectome.tracking as candidate
import fnit.connectome.sift2_mapping as mapping
from fnit.connectome.assignment import build_connectomes
from fnit.connectome.tcksample_precise import sample_streamline_mean_precise


def baseline_module(root, name):
    path = root / "src/fnit/connectome" / (name + ".py")
    spec = importlib.util.spec_from_file_location("fnit.connectome._baseline_" + name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compare_tracks(left, right):
    assert len(left.paths) == len(right.paths)
    for a, b in zip(left.paths, right.paths):
        assert torch.equal(a, b), "path changed"
    for name in ("endpoints", "lengths_mm", "accepted_seeds"):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
    assert left.seeds_attempted == right.seeds_attempted


def old_collect(forward, backward, nf, nb, one_way, keep, total, seeds):
    paths, endpoints, lengths, accepted = [], [], [], []
    for index in keep.tolist():
        path = (forward[index, :nf[index]] if bool(one_way[index]) else
                torch.cat((backward[index, :nb[index]].flip(0), forward[index, 1:nf[index]])))
        paths.append(path)
        endpoints.append(torch.stack((path[0], path[-1])))
        lengths.append(total[index])
        accepted.append(seeds[index])
    return candidate.Tractogram(tuple(paths), torch.stack(endpoints), torch.stack(lengths),
                               None, len(seeds), torch.stack(accepted))


def paired_times(functions, device, repeats):
    for function in functions.values():
        function()
    _sync(device)
    times = defaultdict(list)
    for repeat in range(repeats):
        names = list(functions) if repeat % 2 == 0 else list(functions)[::-1]
        for name in names:
            _sync(device)
            started = time.perf_counter()
            result = functions[name]()
            _sync(device)
            times[name].append(time.perf_counter() - started)
            del result
    return {name: {"seconds": values, "median_seconds": statistics.median(values)}
            for name, values in times.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-root", "fod", "five-tissue", "gmwmi", "fa", "tracks", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--atlas-manifest", type=Path,
                        help="TSV: atlas name, NIfTI path, node count")
    parser.add_argument("--track-metrics", type=Path,
                        help="Fixed-TCK npz: weights and lengths, in TCK order")
    parser.add_argument("--n-seeds", type=int, default=1000)
    parser.add_argument("--baseline-commit", help="Commit of the frozen baseline checkout")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or args.repeats < 1:
        parser.error("this profiler requires CUDA and positive --repeats")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(17e9 / torch.cuda.get_device_properties(device).total_memory,
                                               device)
    baseline = baseline_module(args.baseline_root, "tracking")
    old_mapping = baseline_module(args.baseline_root, "sift2_mapping")
    old_fa = baseline_module(args.baseline_root, "tcksample_precise")
    old_fa._upsample_tracks = old_mapping._upsample_tracks
    fod, affine = _load(args.fod, device)
    five, five_affine = _load(args.five_tissue, device)
    gmwmi, _ = _load(args.gmwmi, device)
    fa, fa_affine = _load(args.fa, device)
    image = nib.load(str(args.five_tissue))
    options = dict(n_seeds=args.n_seeds, batch_size=args.batch_size, seed=0,
                   five_tissue_spacing_mm=image.header.get_zooms()[:3])
    saved_grows, saved_act, events = [], [], defaultdict(list)
    originals = {}
    for name in ("_grow", "_ifod2_arc_probability", "_sample", "_five_tissue_mrtrix",
                 "tracking_sh_precomputed", "_act_structural_step", "_act_seed_direction"):
        originals[name] = getattr(baseline, name)
        def wrapped(*values, _name=name, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = originals[_name](*values, **kwargs)
            end.record()
            events[_name].append((start, end))
            if _name == "_grow":
                saved_grows.append((values[0], result))
            if _name == "_act_seed_direction":
                saved_act.append(result)
            return result
        setattr(baseline, name, wrapped)
    torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    old_tracks = baseline.probabilistic_tractography(fod, affine, five, five_affine, gmwmi, **options)
    _sync(device)
    profile_seconds = time.perf_counter() - started
    profile = {name: {"calls": len(pairs), "cuda_seconds_inclusive":
                      sum(start.elapsed_time(end) for start, end in pairs) / 1000}
               for name, pairs in events.items()}
    for name, function in originals.items():
        setattr(baseline, name, function)
    started = time.perf_counter()
    new_tracks = candidate.probabilistic_tractography(fod, affine, five, five_affine, gmwmi, **options)
    _sync(device)
    candidate_seconds = time.perf_counter() - started
    compare_tracks(old_tracks, new_tracks)
    accepted_streamlines = len(old_tracks.paths)
    # Reuse actual padded propagation buffers; no replay of random sampling.
    collection_arguments = []
    for batch, (_, one_way, _) in enumerate(saved_act):
        seeds, (forward, nf, gf, lf, wf) = saved_grows[2 * batch]
        _, (backward, nb, gb, lb, wb) = saved_grows[2 * batch + 1]
        total = torch.where(one_way, lf, lf + lb)
        # Match the accepted subset by seed coordinates from the complete run.
        keep = ((seeds[:, None] == old_tracks.accepted_seeds[None]).all(-1).any(-1)).nonzero().flatten()
        values = (forward, backward, nf, nb, one_way, keep, total, seeds)
        compare_tracks(old_collect(*values), candidate._collect_tracks(*values))
        collection_arguments.append(values)
    functions = {"baseline": lambda: [old_collect(*v) for v in collection_arguments],
                 "optimized": lambda: [candidate._collect_tracks(*v) for v in collection_arguments]}
    collection_times = paired_times(functions, device, args.repeats)
    scalar_calls = {}
    for name, function in functions.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            function()
            _sync(device)
        scalar_calls[name] = sum(e.count for e in prof.key_averages()
                                 if e.key == "aten::_local_scalar_dense")
    del saved_grows, collection_arguments, functions, old_tracks, new_tracks
    loaded = nib.streamlines.load(str(args.tracks)).tractogram.streamlines
    paths = tuple(torch.as_tensor(np.asarray(path), device=device) for path in loaded)
    before = old_mapping._upsample_tracks(paths, 1)
    after = mapping._upsample_tracks(paths, 1)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    del before, after
    packing_times = paired_times({"baseline": lambda: old_mapping._upsample_tracks(paths, 1),
                                  "optimized": lambda: mapping._upsample_tracks(paths, 1)},
                                 device, args.repeats)
    fa_functions = {"baseline": lambda: old_fa.sample_streamline_mean_precise(paths, fa, fa_affine),
                    "optimized": lambda: sample_streamline_mean_precise(paths, fa, fa_affine)}
    old_values, new_values = (function() for function in fa_functions.values())
    assert torch.equal(old_values, new_values), "precise FA changed"
    fa_times = paired_times(fa_functions, device, args.repeats)
    matrices = {}
    if args.atlas_manifest:
        if not args.track_metrics:
            raise ValueError("--track-metrics is required with --atlas-manifest")
        metrics = np.load(args.track_metrics)
        weights, lengths = (torch.as_tensor(metrics[name], device=device) for name in ("weights", "lengths"))
        endpoints = torch.as_tensor(np.stack([(p[0], p[-1]) for p in loaded]), device=device)
        for line in args.atlas_manifest.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, path, count = line.split("\t")
            atlas, atlas_affine = _load(Path(path), device, torch.int32)
            old = build_connectomes(endpoints, atlas, atlas_affine, weights=weights,
                                    lengths=lengths, fa=old_values, node_count=int(count))
            new = build_connectomes(endpoints, atlas, atlas_affine, weights=weights,
                                    lengths=lengths, fa=new_values, node_count=int(count))
            assert all(torch.equal(old[k], new[k]) for k in old)
            matrices[name] = {"nodes": int(count), "all_four_bitwise_equal": True,
                              "atlas_sha256": _sha256(Path(path))}
    report = {
        "dataset": "OpenNeuro ds004666 sub-01/ses-2mm",
        "baseline_commit": args.baseline_commit,
        "source_sha256": {name: _sha256(Path(module.__file__)) for name, module in
                          (("baseline_tracking", baseline), ("tracking", candidate),
                           ("baseline_mapping", old_mapping), ("mapping", mapping))},
        "input_sha256": {name: _sha256(getattr(args, name)) for name in
                         ("fod", "five_tissue", "gmwmi", "fa", "tracks")},
        "device": str(device), "gpu": torch.cuda.get_device_name(device), "tf32": True,
        "n_seeds": args.n_seeds, "batch_size": args.batch_size,
        "accepted_streamlines": accepted_streamlines,
        "tracking_profile_seconds": profile_seconds,
        "candidate_tracking_seconds": candidate_seconds,
        "profile_note": "Inclusive CUDA events overlap; instrumented vs uninstrumented total is not a speedup comparison.",
        "profile": profile, "tracking_paths_and_metrics_bitwise_equal": True,
        "collection": collection_times, "collection_cpu_scalar_calls": scalar_calls,
        "fixed_tck_streamlines": len(paths), "fixed_tck_points": sum(len(p) for p in paths),
        "packing": packing_times, "precise_fa": fa_times,
        "packing_and_precise_fa_bitwise_equal": True, "matrices": matrices,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
