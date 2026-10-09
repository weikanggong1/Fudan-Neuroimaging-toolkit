"""Independent CPU/CUDA SH controls; no tractography or speed benchmark.

Examples:
  python sh_controls.py --baseline-fod baseline_fod.py --candidate-fod fod.py \
      --fixture real_ifod2_arc_ds004666.npz --device cuda:0 --output sh.json

Optional --benchmark-tool and --tracking-source together use the benchmark's
explicit two-FOD loader; direct mode also loads two independent FOD modules.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import torch


def load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load source: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().reshape(-1).view(np.uint8).tobytes()


def input_state(value: torch.Tensor):
    return str(value.dtype), tuple(value.shape), tuple(value.stride()), str(value.device), raw(value)


def rng_state(device: torch.device, generators: list[torch.Generator]):
    return (raw(torch.get_rng_state()),
            raw(torch.cuda.get_rng_state(device)) if device.type == 'cuda' else None,
            tuple(raw(generator.get_state()) for generator in generators),
            pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()))


def assert_exact(expected: torch.Tensor, actual: torch.Tensor, label: str):
    assert expected.shape == actual.shape, f'{label}: shape mismatch'
    assert expected.dtype == actual.dtype, f'{label}: dtype mismatch'
    assert expected.device == actual.device, f'{label}: device mismatch'
    # byte comparison covers NaN payloads and signed zero, unlike allclose/equal.
    assert raw(expected) == raw(actual), f'{label}: raw bytes differ'


def forward_control(baseline, candidate, data, lmax, generators):
    before = input_state(data)
    rng = rng_state(data.device, generators)
    expected = baseline.tracking_sh_precomputed(data, lmax)
    assert input_state(data) == before, 'baseline mutated input'
    assert rng_state(data.device, generators) == rng, 'baseline mutated RNG'
    actual = candidate.tracking_sh_precomputed(data, lmax)
    assert_exact(expected, actual, f'lmax={lmax},shape={tuple(data.shape)},stride={tuple(data.stride())}')
    assert input_state(data) == before, 'candidate mutated input'
    assert rng_state(data.device, generators) == rng, 'candidate mutated RNG'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-fod', type=Path, required=True)
    parser.add_argument('--candidate-fod', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--baseline-commit', help='Optional verified baseline commit label')
    parser.add_argument('--fixture', type=Path,
                        default=Path(__file__).resolve().parent / 'real_ifod2_arc_ds004666.npz')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--benchmark-tool', type=Path)
    parser.add_argument('--tracking-source', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-cap-bytes', type=int, default=18_000_000_000)
    args = parser.parse_args()
    if (args.benchmark_tool is None) != (args.tracking_source is None):
        parser.error('--benchmark-tool and --tracking-source must be supplied together')
    if args.threads < 1 or args.memory_cap_bytes < 1:
        parser.error('threads and memory cap must be positive')
    for value in [args.baseline_fod, args.candidate_fod, args.fixture,
                  args.benchmark_tool, args.tracking_source]:
        if value is not None and not value.is_file():
            parser.error(f'Missing input: {value}')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('device must be cpu or cuda')
    device_metadata = {'type': device.type}
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; CPU fallback is disabled')
        torch.cuda.set_device(device)
        device = torch.device('cuda', torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        cap = min(args.memory_cap_bytes, int(props.total_memory * .9))
        torch.cuda.set_per_process_memory_fraction(cap / props.total_memory, device)
        # Match probabilistic_tractography's existing policy; no lower precision.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.get_rng_state(device)  # Initialize before protected snapshots.
        torch.cuda.reset_peak_memory_stats(device)
        device_metadata.update(name=props.name, memory_cap_bytes=cap,
                               total_memory_bytes=props.total_memory)
    if args.benchmark_tool is not None:
        tool = load_source('_fnit_sh_controls_benchmark', args.benchmark_tool)
        tracking, fods = tool.load_source_pairs(
            args.tracking_source, args.tracking_source,
            baseline_fod_path=args.baseline_fod,
            candidate_fod_path=args.candidate_fod,
        )
        for variant in ['baseline', 'candidate']:
            assert tracking[variant].tracking_sh_precomputed is fods[variant].tracking_sh_precomputed
        baseline, candidate = fods['baseline'], fods['candidate']
        loader = 'benchmark_load_source_pairs'
    else:
        baseline = load_source('_fnit_sh_controls_baseline_fod', args.baseline_fod)
        candidate = load_source('_fnit_sh_controls_candidate_fod', args.candidate_fod)
        loader = 'independent_direct_fod_modules'
    assert baseline is not candidate, 'FOD modules must be distinct objects'
    assert baseline.tracking_sh_precomputed is not candidate.tracking_sh_precomputed

    generator = torch.Generator().manual_seed(20261009)
    generators = [generator]
    if device.type == 'cuda':
        generators.append(torch.Generator(device=device).manual_seed(20261009))
    controls = [torch.randn((513, 3), generator=generator),
                torch.randn((257, 3), generator=generator) * 1e-19,
                torch.randn((257, 3), generator=generator) * 1e19,
                torch.tensor([[0., 0., 1.], [0., 0., -1.], [1., 0., 0.], [-1., 0., 0.],
                              [0., 1., 0.], [0., -1., 0.], [0., 0., 0.],
                              [1e-10, -1e-10, 1.], [-1e-10, 1e-10, -1.],
                              [-0., 0., 1.], [0., -0., -1.],
                              [float('nan'), 1, 2], [float('inf'), 1, 2],
                              [-float('inf'), -0., 0.]], dtype=torch.float32)]
    with np.load(args.fixture, allow_pickle=False) as real:
        controls.append(torch.from_numpy(real['directions'].copy()))
    cases = points = 0
    widths = []
    for lmax in [0, 2, 4, 6, 8, 10, 12]:
        centres, orders = baseline._tracking_sh_indices(lmax, device)
        coverage = torch.cat((centres, *(index for pair in orders for index in pair)))
        width = (lmax + 1) * (lmax + 2) // 2
        assert torch.equal(coverage.sort().values, torch.arange(width, device=device))
        widths.append(width)
        for original in controls:
            for shape in [(-1, 3), (-1, 1, 3)]:
                contiguous = original.reshape(*shape).to(device)
                # Allocate on target device before slicing, so CUDA preserves strides.
                backing = torch.empty((*contiguous.shape[:-1], 6), dtype=contiguous.dtype, device=device)
                strided = backing[..., ::2]
                strided.copy_(contiguous)
                for data in [contiguous, strided]:
                    forward_control(baseline, candidate, data, lmax, generators)
                    cases += 1
                    points += data.numel() // 3
    assert cases == 140, 'Original 140 controls were not preserved'

    large_controls = []
    for size in [1, 128, 8192, 32768, 131072]:
        data_cpu = torch.randn((size, 3), generator=generator)
        for strided_layout in [False, True]:
            if strided_layout:
                data = torch.empty((size, 6), dtype=torch.float32, device=device)[:, ::2]
                data.copy_(data_cpu)
            else:
                data = data_cpu.to(device)
            forward_control(baseline, candidate, data, 8, generators)
            large_controls.append({'directions': size, 'strided': strided_layout,
                                   'raw_bytes_equal': True})

    empty_controls = []
    for lmax in [0, 8]:
        errors = []
        data = torch.empty((0, 3), dtype=torch.float32, device=device)
        before, rng = input_state(data), rng_state(device, generators)
        for module in [baseline, candidate]:
            try:
                module.tracking_sh_precomputed(data, lmax)
            except Exception as problem:
                errors.append((type(problem).__name__, str(problem)))
            else:
                errors.append(None)
        assert errors[0] == errors[1], f'Empty input behavior differs: {errors}'
        assert input_state(data) == before and rng_state(device, generators) == rng
        empty_controls.append({'lmax': lmax, 'same_exception': True,
                               'exception': errors[0][0] if errors[0] else None})

    gradient_controls = []
    for lmax in [0, 2, 8, 12]:
        original = torch.randn((97, 3), generator=generator)
        for strided_layout in [False, True]:
            values, outputs, gradients = [], [], []
            rng = rng_state(device, generators)
            for module in [baseline, candidate]:
                if strided_layout:
                    value = torch.empty((97, 6), dtype=torch.float32, device=device)[:, ::2]
                    value.copy_(original)
                else:
                    value = original.to(device).clone()
                value.requires_grad_(True)
                before = input_state(value)
                result = module.tracking_sh_precomputed(value, lmax)
                result.square().sum().backward()
                assert input_state(value) == before, 'Gradient path mutated input'
                assert rng_state(device, generators) == rng, 'Gradient path mutated RNG'
                values.append(value)
                outputs.append(result)
                gradients.append(value.grad)
            assert_exact(outputs[0], outputs[1], f'gradient forward lmax={lmax}')
            assert_exact(gradients[0], gradients[1], f'gradient backward lmax={lmax}')
            # Explicitly exercise inference with a requires_grad input as well.
            with torch.no_grad():
                first = baseline.tracking_sh_precomputed(values[0], lmax)
                second = candidate.tracking_sh_precomputed(values[1], lmax)
            assert_exact(first, second, f'no_grad lmax={lmax}')
            assert rng_state(device, generators) == rng
            gradient_controls.append({'lmax': lmax, 'strided': strided_layout,
                                      'forward_raw_bytes_equal': True,
                                      'backward_raw_bytes_equal': True,
                                      'no_grad_raw_bytes_equal': True})

    sample = torch.randn((8192, 3), generator=generator).to(device)
    operator_counts = {}
    for name, module in [('baseline', baseline), ('candidate', candidate)]:
        module.tracking_sh_precomputed(sample, 8)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        # Count Python-dispatched ATen operations, not inclusive CUDA time.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            module.tracking_sh_precomputed(sample, 8)
        operator_counts[name] = {event.key: event.count for event in profile.key_averages()
                                if event.key in ['aten::index_select', 'aten::index_copy_',
                                                 'aten::mul', 'aten::zeros_like',
                                                 'aten::zero_', 'aten::stack', 'aten::cat']}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    report = {
        'scope': 'Exact SH controls only; no tractography or speed benchmark',
        'status': 'passed',
        'baseline_commit': args.baseline_commit or (
            '4f56cc9d5937a766c432743809ab3b5baf11ca78'
            if sha(args.baseline_fod) == 'f82d2174109d05b9f820b6abe98e3792e48ae63e02ace2f2640f10c4c2052aaa' else None),
        'baseline_fod_sha256': sha(args.baseline_fod),
        'candidate_fod_sha256': sha(args.candidate_fod),
        'fixture_sha256': sha(args.fixture),
        'runner_sha256': sha(Path(__file__)),
        'benchmark_tool_sha256': sha(args.benchmark_tool) if args.benchmark_tool else None,
        'tracking_source_sha256': sha(args.tracking_source) if args.tracking_source else None,
        'source_loader': loader,
        'fod_module_identity_distinct': True,
        'sh_function_identity_distinct': True,
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cpu_threads': torch.get_num_threads(),
        'device': device_metadata,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'autocast_enabled': torch.is_autocast_enabled(),
        'original_control_cases': cases,
        'original_evaluated_directions': points,
        'lmax': [0, 2, 4, 6, 8, 10, 12],
        'complete_unique_column_coverage': widths,
        'raw_bytes_equal': True,
        'input_unchanged': True,
        'rng_unchanged': True,
        'large_controls': large_controls,
        'empty_input_existing_behavior': empty_controls,
        'gradient_controls': gradient_controls,
        'host_aten_operator_counts_lmax8_8192': operator_counts,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
    }
    rendered = json.dumps(report, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
