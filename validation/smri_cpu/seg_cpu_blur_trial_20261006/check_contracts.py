"""Bounded blur contracts, not a real-image benchmark or a CPU speed claim."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source))
    import torch
    import channel_batch as trial
    from fnit.synthseg_parc import segment
    from fnit.synthseg_parc.cpu_conv import convolution_slabs
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    initial = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
               torch.backends.mkldnn.enabled, torch.cuda.is_initialized())
    generator = torch.Generator().manual_seed(20261006)
    rows = []
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        for name, dims, strided, cap in (("boundaries",(7,9,11),False,2**28),
                ("many_slabs",(7,9,11),False,33*9*11*4*3),
                ("single_depth",(1,9,11),False,2**28),
                ("strided",(7,9,11),True,2**28),
                ("M32768",(32,32,32),False,2**28)):
            shape=(1,33,*dims)
            original = torch.rand(shape, generator=generator)
            original[:,0].zero_()
            original[:,1].fill_(-0.)
            if strided:
                storage=torch.empty((1,33,dims[0],dims[1],2*dims[2]))
                storage[...,::2].copy_(original)
                original=storage[...,::2]
            input_bits=original.clone().view(torch.int32)
            axis=torch.arange(-1,2,dtype=torch.float32)
            grid=torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"))
            kernel=torch.exp(-grid.square().sum(0)/(2*.5**2))
            kernel=(kernel/kernel.sum()).view(1,1,3,3,3)
            expected=convolution_slabs(original,kernel.expand(33,1,3,3,3),
                                       padding=1,groups=33,maximum_slab_bytes=cap)
            actual=trial.channel_batch_slabs(original,kernel,maximum_slab_bytes=cap)
            different=int(torch.count_nonzero(actual.view(torch.int32)!=expected.view(torch.int32)))
            assert different==0, (name,different,float((actual-expected).abs().max()))
            assert actual.shape==expected.shape and actual.stride()==expected.stride()
            assert actual.is_contiguous() and bool(torch.isfinite(actual).all())
            assert actual.dtype==torch.float32 and actual.device.type=="cpu"
            assert actual.untyped_storage().data_ptr()!=original.untyped_storage().data_ptr()
            assert torch.equal(original.clone().view(torch.int32),input_bits)
            rows.append({"name":name,"shape":list(shape),"input_strided":strided,
                         "bit_different_values":different,"output_stride":list(actual.stride())})
        # Actual Gaussian construction and the eligibility-check call are covered.
        assert torch.equal(trial.blur(original,segment._blur).view(torch.int32),
                           segment._blur(original).view(torch.int32))
    called=[]
    sentinel=object()
    def fallback(value):
        called.append(value)
        return sentinel
    guard_input=torch.empty((1,33,2,3,4))
    assert trial.blur(guard_input,fallback) is sentinel  # grad-enabled caller
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=True):
        assert trial.blur(guard_input,fallback) is sentinel
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        assert trial.blur(torch.empty((2,33,2,3,4)),fallback) is sentinel
        assert trial.blur(torch.empty((1,69,2,3,4)),fallback) is sentinel
        assert trial.blur(torch.empty((1,33,2,3,4),dtype=torch.float64),fallback) is sentinel
        assert trial.blur(torch.empty((1,33,2,3,4),device="meta"),fallback) is sentinel
        with patch("fnit.synthseg_parc.cpu_conv.cpu_autocast_enabled",return_value=True):
            assert trial.blur(guard_input,fallback) is sentinel
        with patch.object(trial.F,"conv3d",side_effect=RuntimeError("declared conv failure")):
            try:
                trial.blur(guard_input,fallback)
            except RuntimeError as exc:
                assert str(exc)=="declared conv failure"
            else:
                raise AssertionError("exception was swallowed")
    assert len(called)==7
    final=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
           torch.backends.mkldnn.enabled,torch.cuda.is_initialized())
    assert final==initial and not final[-1]
    report={"schema":"fnit_seg_cpu_channel_batch_contracts/v1","status":"passed",
        "scope":"Synthetic CPU-only value/alias/slab/guard contracts; not a real MRI or speed benchmark",
        "torch_version":torch.__version__,"torch_parallel_info":torch.__config__.parallel_info(),
        "threads":torch.get_num_threads(),"cases":rows,"fallback_calls":len(called),
        "initial_flags":list(initial),"final_flags":list(final),
        "cuda_initialized":torch.cuda.is_initialized(),
        "helper_sha256":sha(Path(__file__).with_name("channel_batch.py")),
        "contract_sha256":sha(__file__),
        "source_files":{name:sha(args.source/"fnit/synthseg_parc"/name)
                        for name in ("cpu_conv.py","segment.py")}}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"cases":len(rows),"torch_version":torch.__version__}))


if __name__=="__main__":
    main()
