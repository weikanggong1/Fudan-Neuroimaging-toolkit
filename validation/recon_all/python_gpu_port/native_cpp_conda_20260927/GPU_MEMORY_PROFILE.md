# GPU memory: exact-input SynthSeg in the Conda recon-all profile

## Scope and reproducible input

On gpucw1 (H100, PyTorch 2.5.1/CUDA 11.8), the already completed Conda recon-all run left PID 67057 with **24,634 MiB on GPU1** during CPU surface commands. Its `mri/orig.mgz` is 256 × 256 × 256 uint8 at 1 mm. The baseline run completed before this memory change. We replayed its exact SynthSeg input and external weights in **separate processes**, without rerunning recon-all or modifying the comparison subject. The Python call was `SynthSeg(weights=weights_dir, device='cuda:1', threads=4)(orig_mgz, keep_geometry=True, color_lut=lut)`. The corresponding FreeSurfer stage is `mri_synthseg --i orig.mgz --o synthseg.rca.mgz --vol synthseg.vol.csv --threads 4 --keepgeom --addctab --cpu` with the same official 2.0 weights. That command was not rerun for this memory audit; its numerical comparison is documented elsewhere.

The diagnostic script [`profile_synthseg_gpu_memory.py`](profile_synthseg_gpu_memory.py) instruments preprocessing, both FP32 U-Net forwards, postprocessing, and volume calculation without changing the production module. It records PyTorch `max_memory_allocated` and `max_memory_reserved`, plus `nvidia-smi` samples at function boundaries. `nvidia-smi` values are sampled and are not guaranteed continuous peaks. The process set cuDNN deterministic mode and disabled TF32 for SynthSeg, matching the existing fixed voxel-parity path; the rest of FNIT retains its existing TF32 defaults. FP16/BF16 were not used.

## Results

| Exact-input variant | Peak allocated | Peak reserved | Largest `nvidia-smi` sample | Result return | Hard segmentation / 32 volumes + TIV |
|---|---:|---:|---:|---:|---|
| Original, cuDNN benchmark on | 17,010 MiB | 21,310 MiB | 21,846 MiB | 11.081 s | Reference |
| Original, cuDNN benchmark off | 17,010 MiB | 21,310 MiB | 21,846 MiB | 10.189 s | Exact vs reference |
| Production patch, benchmark on | 17,010 MiB | **18,616 MiB** | **19,152 MiB** | 11.170 s | Exact vs reference |

The baseline and patched segmentation byte hash was `35fee7ff85db832c8698daf14275b50bfe97adbfcc9f13e3611c08ec34a3a435`. All 32 per-structure soft volumes and total intracranial volume matched exactly; the near-tie count was 1 in every run. The patched segmentation also matched the completed baseline E2E `mri/synthseg.rca.mgz` voxel-for-voxel by hash, and its float32-formatted 33 CSV numbers matched the baseline `stats/synthseg.vol.csv` in all 33 columns. No timing gain is claimed from these one-off measurements. Detailed counters and metric values are in [`gpu_memory_true.json`](gpu_memory_true.json), [`gpu_memory_false.json`](gpu_memory_false.json), and [`gpu_memory_production_patch.json`](gpu_memory_production_patch.json).

The first U-Net forward reached 14,834 MiB allocated / 16,700 MiB reserved. The second reached 17,010 / 21,308 MiB in the baseline. Clearing *unused* PyTorch cache after the first blur, before the second forward, kept the same 17,010 MiB allocated peak but lowered the reserved peak to 18,616 MiB. The patched sample of 19,152 MiB is 18.70 GiB, or approximately 20.08 decimal GB; report both units when applying a nominal “20 GB” budget.

After the original `SynthSeg` result returned and its object was garbage-collected, PyTorch reported **0 MiB allocated but 21,310 MiB reserved**; `nvidia-smi` still reported 21,846 MiB. `torch.cuda.empty_cache()` lowered reserved to 0 and `nvidia-smi` to 536 MiB for the process's CUDA context. This confirms that the CPU-stage residency was predominantly retained allocator cache, although other preceding GPU stages also contributed to the original E2E 24,634 MiB. Turning cuDNN benchmark off did not reduce memory on this input.

## Production change and limits

- [`segment.py`](../../../../src/fnit/synthseg_parc/segment.py) now releases unused cache after the first blurred posterior, only on the full GPU flip-ensemble path, before the second U-Net forward. It preserves FP32 tensors, output values, TF32 behavior, and model settings.
- [`native_free.py`](../../../../src/fnit/recon_all/native_free.py) now clears the GPU allocator before and after its SynthSeg stage. This prevents prior GPU-stage cache from stacking with the network and frees residency before the long CPU native commands. CPU calls do not touch CUDA cache.

The patched standalone SynthSeg replay establishes output identity and a lower isolated peak. Three focused standalone SynthSeg/tie tests passed in the target Conda environment; `py_compile` and `git diff --check` passed. A **fresh v2 full recon-all run did complete** (30 stages, `exit=0`, 2648.95 s). Its GPU1 PID sampler recorded a **20,824 MiB** maximum during SynthSeg and 2,256 MiB after it. Sampling began after the start of reconstruction and was periodic, so this is an observed maximum rather than a guaranteed continuous peak. The v2 strict output comparison is separate from this memory test; its numerical failures remain. A future multi-subject concurrency budget must add the peaks of simultaneously active subjects on each device.

## Upstream residency and process-isolation candidate

The standalone [`profile_pre_synthseg_residency.py`](profile_pre_synthseg_residency.py) replay used the same T1 and v2 `nu.mgz` in new temporary subjects. It recorded PyTorch allocated/reserved memory and per-PID `nvidia-smi` at stage boundaries. `outside_torch_allocator_mib` is `nvidia-smi - torch.cuda.memory_reserved`, an accounting difference rather than a named library allocation. All memory values below are MiB; time is measured from the diagnostic script's first mark. The source-built FreeSurfer programs were not involved in this prefix replay.

| Boundary | Default parent process | SynthMorph in short-lived child |
|---|---:|---:|
| Fresh CUDA context | 448 | 448 |
| After input/Talairach and `empty_cache()` | 2,254 | **536** |
| After T1 normalization and `empty_cache()` | 2,254 | 538 |
| After brainmask and `empty_cache()` | 2,254 | 538 |
| First SynthSeg U-Net forward | 18,650 | 17,238 |
| Second SynthSeg U-Net forward | 20,820 | **19,152** |
| Largest sampled SynthSeg residency | 20,824 | **19,154** |
| After SynthSeg and `empty_cache()` | 2,256 | 538 |
| Diagnostic elapsed from first mark | 88.36 s | 89.09 s |

The input-only replay narrowed the residual to SynthMorph: `nvidia-smi - torch.reserved` was 448 MiB in a fresh CUDA context, 526 after conform, 536 after SynthStrip, and **1,820 after SynthMorph**. The production runner does not keep the input-chain models in module globals; its returned object contains paths and timings. SynthMorph's CUDA execution accounts for the additional observed process residency, but the experiment does not identify a specific CUDA library allocation. The default, substage, and isolated records are [`gpu_prefix_residency.json`](gpu_prefix_residency.json), [`gpu_input_stage_residency.json`](gpu_input_stage_residency.json), and [`gpu_prefix_isolated.json`](gpu_prefix_isolated.json). The isolated full-prefix replay took 93.64 s wall time including Python startup, recorded in [`gpu_prefix_isolated.time`](gpu_prefix_isolated.time). One run per arm on a shared GPU is insufficient for a timing-speed claim.

Both full-prefix arms yielded the same SynthSeg hard-label SHA256 `35fee7ff85db832c8698daf14275b50bfe97adbfcc9f13e3611c08ec34a3a435`, all 32 soft volumes, and total intracranial volume. On the *frozen v2 SynthStrip file*, a child process that copied the parent's cuDNN benchmark, deterministic and TF32 flags generated **byte-identical** `talairach.xfm` and affine LTA; see [`synthmorph_subprocess_same_input.json`](synthmorph_subprocess_same_input.json). A child launched with default cuDNN flags differed by up to 0.01048 in the XFM matrix, so those inherited settings are required; see [`synthmorph_subprocess_default_flags.json`](synthmorph_subprocess_default_flags.json). Separate fresh input-chain replays had identical `orig` voxels but different SynthStrip voxel hashes; their XFM matrices differed by up to 0.02003. See [`gpu_input_direct_paired.json`](gpu_input_direct_paired.json) and [`gpu_input_isolated_paired.json`](gpu_input_isolated_paired.json). These unpaired transforms cannot establish complete-subject identity.

Disabling the cuDNN v8 plan cache with `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=-1` did **not** remove the 1,820 MiB outside-allocator residency and increased post-input allocator reservation from 434 to 1,730 MiB. It produced the same SynthSeg labels and volumes; its sampled maximum was 20,456 MiB and diagnostic time 90.82 s. This option is **not recommended** for the runner. PyTorch 2.5.1 defines the cache setting in [its source](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cudnn/Conv_v8.cpp); the observed residual has a different cause. See [`gpu_prefix_no_cache.json`](gpu_prefix_no_cache.json).

**Recommendation:** retain SynthMorph subprocess isolation as an experimental candidate. Its 19,154 MiB sampled parent-process maximum is 18.70 GiB (about 20.08 decimal GB), close to but slightly above a strict 20.00 GB target. It has not been connected to the production runner or tested in a fresh full 138-file recon-all comparison. The v2 end-to-end result remains the current validated implementation.

## Replay command

The script and the three JSON records in this directory are the audit artifacts. On gpucw1, after copying the script to a work directory, run one process per variant:

```bash
root=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_cpp_conda_20260927
subject=/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/reconall_conda_cpp_e2e_20260927/subjects/sub01
"$root/conda_env/bin/python" "$root/profile_synthseg_gpu_memory.py" \
  --image "$subject/mri/orig.mgz" --weights-dir "$root/weights" \
  --lut "$root/assets/FreeSurferColorLUT.txt" --device cuda:1 \
  --benchmark true --output "$root/logs/gpu_memory_production_patch.json"
```
