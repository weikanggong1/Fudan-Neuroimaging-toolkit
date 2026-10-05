"""Finite observer contracts: no MRI, no fitted model, no performance claim."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import torch
    import nibabel as nib
    from fnit.synthseg_parc import synthseg, segment, cpu_conv, postprocess, preprocess
    spec = importlib.util.spec_from_file_location("profile_one", args.worker)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    network = segment.SegmentUNet().eval()
    model = type("Holder", (), {"segmenter": type("Child", (), {"model": network})()})()
    original_flags = worker.flags(torch)
    names = ("sum", "argmax", "clone", "__itruediv__", "__setitem__")
    methods = {name: getattr(torch.Tensor, name) for name in names}
    original_forward = cpu_conv.CPUInferenceConv3d.forward
    observer = worker.Observer()
    passed = []
    with torch.inference_mode():
        skip = torch.arange(24*8**3, dtype=torch.float32).reshape(1,24,8,8,8)
        value = torch.arange(48*4**3, dtype=torch.float32).reshape(1,48,4,4,4)
        assert segment.cpu_join_allowed(network, skip, value)
        expected = segment.join_nearest_cpu(skip,value)
        image = torch.arange(5*6*7, dtype=torch.float32).reshape(1,1,5,6,7) / 10
        kernel = torch.ones(1,1,3,3,3) / 27
        expected_conv = cpu_conv.convolution_slabs(image, kernel, padding=1)
        try:
            worker.install(observer, torch, nib, (synthseg, segment, cpu_conv, postprocess, preprocess), model)
            assert segment.cpu_join_allowed(network, skip, value)
            output = segment.join_nearest_cpu(skip,value)
            assert torch.equal(output, expected) and output.stride()==expected.stride()
            assert not any(item._forward_hooks or item._forward_pre_hooks for item in network.modules())
            passed.append("join_guard_still_eligible_no_hooks_same_values_strides")
            assert torch.equal(cpu_conv.convolution_slabs(image,kernel,padding=1),expected_conv)
            assert any(row['name']=='slab_output_copy' for row in observer.rows)
            passed.append("original_slab_arithmetic_values_equal_copy_observer_live")
            tensor = torch.arange(12, dtype=torch.float32).reshape(3,4) + 1
            original_tensor = tensor.clone()
            with observer.span("postprocess"):
                cloned = tensor.clone()
                total = tensor.sum(0,keepdim=True)
                index = tensor.argmax(0)
                returned = tensor.__itruediv__(total)
            expected_total = methods['sum'](original_tensor,0,keepdim=True)
            assert torch.equal(total, expected_total)
            assert torch.equal(index, methods['argmax'](original_tensor,0))
            assert torch.equal(cloned, original_tensor)
            assert returned is tensor and torch.equal(tensor,original_tensor / expected_total)
            passed.append("sum_argmax_clone_and_inplace_divide_delegate_once_no_alias_change")
            output_image = nib.Nifti1Image(torch.ones(3,4,5).numpy(),torch.eye(4).numpy())
            with observer.span("preprocess"):
                assert output_image.get_fdata().shape==(3,4,5)
            assert any(row['name']=='nifti_data_decode' for row in observer.rows)
            passed.append("nifti_get_fdata_observer_at_actual_base_class")
            try:
                with observer.span("failure_probe"):
                    raise RuntimeError("finite observer probe")
            except RuntimeError:
                pass
            assert not observer.stack and all(row['exclusive_seconds']>=0 for row in observer.rows)
            passed.append("nested_clock_and_exception_unwind")
        finally:
            observer.restore()
    assert worker.flags(torch)==original_flags
    assert all(getattr(torch.Tensor,n) is original for n,original in methods.items())
    assert cpu_conv.CPUInferenceConv3d.forward is original_forward
    passed.append("all_patches_and_global_flags_restore")
    calibration=worker.Observer()
    start=time.perf_counter()
    for _ in range(10000):
        with calibration.span("noop"):
            pass
    calibration_seconds=time.perf_counter()-start
    report={"schema":"fnit_synthseg_cpu_profile_observer_contract/v1", "scope":"finite contracts, no MRI or fitted-model inference",
        "worker_sha256":worker.sha(args.worker),"status":"passed","passed_contracts":passed,
        "event_count":len(observer.rows),"flags":original_flags,
        "noop_10000_observers_seconds":calibration_seconds,
        "noop_mean_seconds":calibration_seconds/10000,
        "calibration_limitation":"No shape/flags metadata cost; a bookkeeping lower-bound estimate, not overhead subtraction for MRI."}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
