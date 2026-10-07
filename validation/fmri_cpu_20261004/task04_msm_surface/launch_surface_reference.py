"""Launch the existing independent original surface reference in an owned run.

The binding is private. It holds the fixed original helper, image, templates,
same-source completed volumes/reconstruction, and existing license path.
No FNIT module is imported and no original algorithm is replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int, choices=(1, 8))
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    for item in binding["helpers"]:
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Pinned independent original helper differs")
    output = args.output_dir
    subprocess.run([
        binding["python"], binding["surface_reference"],
        "--inputs-json", binding["inputs_json"],
        "--output-root", str(output / "results"),
        "--work-root", str(output / "work"),
        "--container-image", binding["container_image"],
        "--templateflow-dir", binding["templateflow"],
        "--fs-license", binding["fs_license"],
        "--singularity", binding["singularity"],
        "--threads", str(args.threads), "--msm-threads", str(args.threads),
    ], check=True)
    report = json.loads((output / "results/run.public.json").read_text())
    if report.get("validation_complete") is not True:
        raise RuntimeError("Independent original chain did not finish validation")


if __name__ == "__main__":
    main()
