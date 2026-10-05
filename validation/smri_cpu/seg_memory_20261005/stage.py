"""Private real decoder checkpoint capture and same-input CPU copy benchmark."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    array = value.detach().numpy()
    h = hashlib.sha256()
    # Preserve C order without a second multi-GB complete tensor allocation.
    for channel in range(array.shape[1]):
        h.update(array[:, channel].tobytes(order="C"))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("capture", "baseline", "candidate"), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--array", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--helper", type=Path)
    p.add_argument("--repeats", type=int, default=4)
    args = p.parse_args()
    os.umask(0o077)
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    source_files = {name: file_sha(args.source / "fnit/synthseg_parc" / name)
                    for name in ("model.py", "segment.py", "cpu_conv.py", "pipeline.py",
                                 "postprocess.py", "synthseg.py")}
    expected = {"model.py": "292428207da96cea837919e73b3fa455dc533b5b9e7fb1bca02e3729a92cde52",
                "segment.py": "ca51439f15ea9b792f2eb4bfb7bdd6cd7f298522e03a5917dda7ed8c9438c441",
                "cpu_conv.py": "78a0fac2300a45a01ff6790d24218f825ebc21a21f2033a8e6b8c58ebb60dc47",
                "pipeline.py": "030095b124a3556f0508359317ee15108fbb2a3d45a803d80c28350d18776633",
                "postprocess.py": "5ac0a53745e972563302ebcde2f078d5fb3e6eb7607ec92e59f67d85adae8936",
                "synthseg.py": "3f0e91ff0bfa83ab8b3742a27ccecabcad83ddb650110d7347e5e4fea42ab62f"}
    assert source_files == expected, "producer is not the qualified SynthSeg baseline"
    sys.path.insert(0, str(args.source))
    import fnit.synthseg_parc.segment as segment
    assert Path(segment.__file__).resolve() == (args.source / "fnit/synthseg_parc/segment.py").resolve()
    report = {"schema": "fnit_synthseg_cpu_decoder_copy/v1", "mode": args.mode,
              "source_files": source_files, "worker_sha256": file_sha(__file__),
              "hostname": os.uname().nodename, "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
              "torch_interop_threads": torch.get_num_interop_threads(),
              "thread_environment": {key: os.environ.get(key) for key in
                  ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
              "dtype": "float32", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    args.report.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.mode == "capture":
        assert not args.checkpoint.exists(), "capture is immutable"
        args.checkpoint.mkdir(parents=True, mode=0o700)
        assert file_sha(args.input) == "73e3866d4e54f9cb253868daab4bf90303a97bc193e8bda21e2e60c53a5dea21"
        array = np.load(args.array)
        assert array.shape == (192, 224, 256) and array.dtype == np.float32
        assert hashlib.sha256(array.tobytes()).hexdigest() == "d34b6799879799e9156a0a0a2c7a977dd084adb82c8847b883adfeaf6d3769ce"
        from fnit import SynthSeg
        runner = SynthSeg(weights=args.weights, device="cpu", threads=8)
        weight = args.weights / "synthseg_2.0.h5"
        report.update(input_sha256=file_sha(args.input), array_sha256=file_sha(args.array),
                      prepared_values_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                      weight_sha256=file_sha(weight))
        interpolate = F.interpolate

        class CaptureComplete(Exception):
            pass

        def copy_source(value, *positional, **keywords):
            if tuple(value.shape) == (1, 48, 96, 112, 128):
                np.save(args.checkpoint / "value.npy", value.detach().numpy())
            return interpolate(value, *positional, **keywords)

        def copy_skip(_layer, positional):
            joined = positional[0]
            assert tuple(joined.shape) == (1, 72, 192, 224, 256)
            np.save(args.checkpoint / "skip.npy", joined[:, :24].detach().numpy())
            raise CaptureComplete()

        from unittest.mock import patch
        handle = runner.segmenter.model.up[3].conv0.register_forward_pre_hook(copy_skip)
        start = time.perf_counter()
        try:
            with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False), \
                    patch.object(F, "interpolate", side_effect=copy_source):
                runner.segmenter.model(torch.from_numpy(array)[None, None])
            raise RuntimeError("capture stop was not reached")
        except CaptureComplete:
            report.update(status="partial_capture_complete", capture_seconds=time.perf_counter() - start,
                          scope="first original CNN stopped before final decoder conv0; no final prediction")
        finally:
            handle.remove()
        report["checkpoint_files"] = {name: {"sha256": file_sha(args.checkpoint / name),
                                             "bytes": (args.checkpoint / name).stat().st_size}
                                      for name in ("skip.npy", "value.npy")}
        (args.checkpoint / "manifest.private.json").write_text(json.dumps(report, indent=2) + "\n")
    else:
        manifest = json.loads((args.checkpoint / "manifest.private.json").read_text())
        assert manifest["status"] == "partial_capture_complete" and manifest["source_files"] == expected
        for name, info in manifest["checkpoint_files"].items():
            assert file_sha(args.checkpoint / name) == info["sha256"]
        start = time.perf_counter()
        skip = torch.from_numpy(np.load(args.checkpoint / "skip.npy"))
        value = torch.from_numpy(np.load(args.checkpoint / "value.npy"))
        report["load_seconds"] = time.perf_counter() - start
        assert skip.is_contiguous() and value.is_contiguous()
        if args.mode == "candidate":
            spec = importlib.util.spec_from_file_location("fnit.synthseg_parc.cpu_join", args.helper)
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            network = segment.SegmentUNet().eval()

            def function(skip, value):
                assert helper.cpu_join_allowed(network, skip, value)
                return helper.join_nearest_cpu(skip, value)

            report["helper_sha256"] = file_sha(args.helper)
            report["qualification_guard_in_operation_timer"] = True
        else:
            function = lambda skip, value: torch.cat((skip, F.interpolate(value, scale_factor=2, mode="nearest")), 1)
        report["shape"] = list(skip.shape)
        report["checkpoint_files"] = manifest["checkpoint_files"]
        seconds, identities = [], []
        before = (tensor_sha(skip), tensor_sha(value))
        with torch.inference_mode():
            for _ in range(args.repeats):
                start = time.perf_counter()
                output = function(skip, value)
                seconds.append(time.perf_counter() - start)
                identities.append({"values_sha256": tensor_sha(output), "shape": list(output.shape),
                                   "stride": list(output.stride()), "contiguous": output.is_contiguous(),
                                   "independent": output.data_ptr() not in (skip.data_ptr(), value.data_ptr())})
                del output
        assert (tensor_sha(skip), tensor_sha(value)) == before
        assert all(item == identities[0] for item in identities)
        report.update(status="stage_complete", operation_seconds=seconds, identities=identities,
                      scope="saved real final decoder nearest+cat only; not complete network")
    report["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "mode", "max_rss_kib")}), flush=True)


if __name__ == "__main__":
    main()
