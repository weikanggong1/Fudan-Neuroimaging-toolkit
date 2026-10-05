"""Prepare full old/new GPU API requests; this script never starts a GPU job."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cpuset", required=True, help="Coordinator assigned GPU-host CPU affinity")
    args = parser.parse_args()
    if not args.device.startswith("cuda"):
        raise ValueError("GPU acceptance requires an explicit CUDA device")
    affinity = [int(value) for value in args.cpuset.split(",")]
    if len(affinity) != 8 or len(set(affinity)) != 8:
        raise ValueError("Assign eight distinct host CPUs for the complete parallel API")
    manifest = json.loads(args.manifest.read_text())
    args.output_root.mkdir(parents=True, exist_ok=False)
    commands = []
    for case in manifest["cases"]:
        for backend, source in (("baseline", args.baseline_root), ("candidate", args.candidate_root)):
            output = args.output_root / case["id"] / backend
            output.mkdir(parents=True)
            request = {"case": case, "adapter": case["adapter"],
                       "adapter_root": str(args.candidate_root.resolve()),
                       "source_root": str(source.resolve()), "output_dir": str(output),
                       "threads": 8, "affinity": affinity, "device": args.device,
                       "backend": backend, "repetitions": 1, "warmup": False}
            request_path = output / "request.private.json"
            result_path = output / "api-result.private.json"
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            commands.append([args.python, str(args.candidate_root / "tools/benchmark_multimodal_cpu.py"),
                             "worker", "--request", str(request_path), "--result", str(result_path)])
    record = {"scope": "Complete three actual registration APIs, all native vertices/configured iterations; old/new source roots separately imported",
              "device": args.device, "cpu_threads": 8, "affinity": affinity,
              "gpu_dispatch": "Coordinator alone acquires GPU lock and chooses process memory cap; requests are not executed here",
              "commands": commands}
    (args.output_root / "commands.private.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"requests": len(commands), "GPU_jobs_started": False}))


if __name__ == "__main__":
    main()
