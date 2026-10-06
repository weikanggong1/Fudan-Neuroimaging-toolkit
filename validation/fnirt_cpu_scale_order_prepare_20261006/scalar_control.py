"""One current-moving cached-state scalar control; no registration/sampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time
from types import SimpleNamespace

from contracts import bound, check_bindings, check_freeze, file_paths, fixed_header, git_head, require, source_baselines, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--canonical-main-commit', required=True)
    parser.add_argument('--approved-scale-control', action='store_true')
    args = parser.parse_args()
    require(args.approved_scale_control, 'explicit finite scalar-control execution approval required')
    plan = json.loads((args.workspace / 'PLAN.public.json').read_text())
    require(args.workspace.resolve() == (args.root / 'workspaces' / plan['canonical_leaf_proposed']).resolve()
            and args.output.resolve() == (args.root / 'runs' / plan['canonical_leaf_proposed'] / 'scalar').resolve(), 'wrong unique canonical scalar leaf')
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    report = {'scope': 'current_saved_moving_accumulation_contract', 'status': 'started',
              'control_completed': False, 'production_integration_accepted': False,
              'same_input_context_as_prior_moving_substitution': False,
              'does_not_explain_prior_projection_scale_error': True,
              'native_samepoint_input_caches_established': False,
              'calls': {'NPZ_member_restore': 0, 'baseline_scalar_pair': 0, 'scalar_FP32_product_pair': 0,
                        'serial_reduction_pair': 0, 'native_saved_g_tail8_read': 0},
              'prohibited_calls': {'sampler': 0, 'coordinates': 0, 'Jte': 0, 'bending': 0, 'H': 0,
                                   'diag': 0, 'PCG': 0, 'SCG': 0, 'native_program': 0, 'GPU': 0, 'raw_MRI': 0, 'full_FNIRT': 0},
              'gates': {}, 'operands_before': {}, 'operands_after': {}, 'source_inputs_unchanged': False,
              'flags_unchanged': None, 'postcheck_errors': [], 'native_scale_comparison_is_posthoc_only': True}
    expected, torch, np, saved = None, None, None, {}
    before, freeze_before, flags_before, header_before = None, None, None, None
    def interrupted(signum, frame):
        raise TimeoutError('finite worker received termination signal')
    signal.signal(signal.SIGTERM, interrupted)
    def scalar(value):
        value = float(value)
        require(math.isfinite(value), 'nonfinite scalar')
        return {'value': value, 'FP64_little_endian_hex': struct.pack('<d', value).hex()}
    def flags():
        return {'cuda_initialized': bool(torch.cuda.is_initialized()), 'cudnn_tf32': bool(torch.backends.cudnn.allow_tf32),
                'matmul_tf32': bool(torch.backends.cuda.matmul.allow_tf32), 'grad_enabled': bool(torch.is_grad_enabled()),
                'default_dtype': str(torch.get_default_dtype()), 'cpu_autocast_enabled': bool(torch.is_autocast_enabled('cpu')),
                'cuda_autocast_enabled': bool(torch.is_autocast_enabled('cuda')), 'threads': int(torch.get_num_threads()),
                'interop_threads': int(torch.get_num_interop_threads())}
    def mapped_fsl():
        return sorted({Path(line.split()[-1]).name for line in Path('/proc/self/maps').read_text().splitlines()
                       if len(line.split()) >= 6 and ('/apps/FSL/' in line.split()[-1] or Path(line.split()[-1]).name.startswith('libfsl'))})
    def value_hash(value):
        value = value.detach().numpy() if isinstance(value, torch.Tensor) else value
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    def scalar_gate(name, wanted, actual):
        row = {'reference': wanted, 'actual': scalar(actual)}
        row['bitexact'] = wanted['FP64_little_endian_hex'] == row['actual']['FP64_little_endian_hex']
        report['gates'][name] = row
        require(row['bitexact'], 'first baseline mismatch: ' + name)
    code = 1
    try:
        expected = json.loads((args.workspace / 'EXPECTED.public.json').read_text())
        report['expected_source'] = bound(args.workspace / 'EXPECTED.public.json')
        before, failed = check_bindings(args.root, expected)
        freeze_before, bad = check_freeze(args.workspace)
        report['bindings_before'], report['harness_before'] = before, freeze_before
        require(not failed and not bad, 'immutable source/input/freeze precondition failed')
        head = git_head(args.root / 'repo')
        require(head == args.canonical_main_commit, 'declared current canonical main mismatch')
        report['canonical_main_commit'] = head
        paths = file_paths(args.root, expected)
        stage1 = json.loads(paths['stage1_summary'].read_text())
        moving = json.loads(paths['moving_control_summary'].read_text())
        require(stage1['accepted_checkpoint'] and moving['control_completed'], 'accepted earlier state missing')
        require(moving['arms']['current_saved_moving']['FSL_order_scale_full_g'] == expected['baseline_scale_full_g'],
                'current-moving baseline metadata mismatch')
        require(stage1['checkpoint']['state_scalars']['count'] == expected['baseline_count'], 'count metadata mismatch')
        header_before = fixed_header(paths['official_fixed'], expected['fixed_header'])
        report['header_before'] = header_before
        env = args.root / 'envs/default'
        require(Path(sys.prefix).resolve() == env.resolve()
                and Path(sys.executable).resolve() == (env / 'bin/python').resolve(), 'wrong actual interpreter/prefix')
        require(bound(Path(sys.executable).resolve()) == expected['runtime_required']['actual_interpreter_identity'],
                'actual interpreter bytes changed')
        require(hashlib.sha256(str(Path(sys.prefix).resolve()).encode()).hexdigest() == expected['runtime_required']['actual_prefix_path_sha256'], 'actual prefix fingerprint changed')
        import numpy as np_module
        import torch as torch_module
        np, torch = np_module, torch_module
        require(np.__version__ == expected['runtime_required']['numpy'] and torch.__version__ == expected['runtime_required']['torch'],
                'scientific dependency version changed')
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        require(sorted(os.sched_getaffinity(0)) == [32,36,40,44,48,52,56,60], 'CPU8 physical affinity mismatch')
        require(resource.getrlimit(resource.RLIMIT_AS)[0] == 8_000_000_000, 'declared address-space bound missing')
        flags_before = flags()
        report['flags_before'], report['FSL_DSO_before'] = flags_before, mapped_fsl()
        require(not flags_before['cuda_initialized'] and not flags_before['cpu_autocast_enabled']
                and not flags_before['cuda_autocast_enabled'] and not report['FSL_DSO_before'], 'unexpected CUDA/autocast/FSL state')
        functions, ast_identity = source_baselines(args.root, {'torch': torch})
        require(ast_identity == expected['scalar_AST'], 'scalar statements changed')
        report['scalar_AST'] = ast_identity
        with torch.no_grad():
            with np.load(paths['checkpoint'], allow_pickle=False) as checkpoint:
                require(set(checkpoint.files) == set(stage1['checkpoint']['arrays']), '68-member checkpoint schema changed')
                for name, record in expected['arrays_to_restore'].items():
                    source = checkpoint[name]
                    report['calls']['NPZ_member_restore'] += 1
                    require([int(v) for v in source.shape] == record['shape'] and str(source.dtype) == record['dtype']
                            and bool(np.isfinite(source).all()), 'checkpoint member schema: ' + name)
                    require(value_hash(source) == record['logical_value_sha256'], 'checkpoint member bits: ' + name)
                    source = source.copy(order='C')
                    tensor = torch.empty_strided(tuple(record['shape']), tuple(v // source.itemsize for v in record['strides']),
                                                dtype=torch.from_numpy(source).dtype, device='cpu')
                    tensor.copy_(torch.from_numpy(source))
                    require([int(v * source.itemsize) for v in tensor.stride()] == record['strides'], 'restored stride changed')
                    saved[name] = tensor
                    report['operands_before'][name] = value_hash(tensor)
            require(len({v.untyped_storage().data_ptr() for v in saved.values()}) == 4, 'restored members alias')
            fixed, residual, mask = saved['fixed'], saved['state_residual'], saved['state_mask']
            count = int(mask.sum())
            require(count == expected['baseline_count'] and fixed.dtype == residual.dtype == torch.float32
                    and mask.dtype == torch.bool and saved['scale'].dtype == torch.float64, 'scalar input contract')
            weight = mask.to(torch.float32)
            report['calls']['scalar_FP32_product_pair'] += 1
            scale_product, square_product = functions['shared_FP32_products'](SimpleNamespace(fixed=fixed), residual, weight, weight)
            require(scale_product.dtype == square_product.dtype == torch.float32, 'FP32 products changed')
            saved['scale_product'], saved['square_product'], saved['weight'] = scale_product, square_product, weight
            for name in ['scale_product', 'square_product', 'weight']:
                report['operands_before'][name] = value_hash(saved[name])
            report['calls']['baseline_scalar_pair'] += 1
            baseline_SSD = functions['baseline_ssd'](square_product, count)
            baseline_half_scale = functions['baseline_scale_half'](scale_product, count)
            baseline_scale = 2 * baseline_half_scale
            scalar_gate('baseline_SSD', expected['baseline_SSD'], baseline_SSD)
            scalar_gate('baseline_scale_full_g', expected['baseline_scale_full_g'], baseline_scale)
            report['baseline_passed_before_candidate'] = True
            product_np, squares_np, mask_np = scale_product.numpy(), square_product.numpy(), mask.numpy()
            dot_sum, SSD_sum = 0.0, 0.0
            report['calls']['serial_reduction_pair'] += 1
            for z in range(24):
                for y in range(28):
                    for x in range(24):
                        if bool(mask_np[x,y,z]):
                            dot_sum += float(product_np[x,y,z])
                            SSD_sum += float(squares_np[x,y,z])
            source_group_scale = (2.0 / float(count)) * (-dot_sum)
            existing_group_scale = 2.0 * (-dot_sum / float(count))
            serial_SSD = SSD_sum / float(count)
            report['serial_candidates'] = {'dot_sum': scalar(dot_sum), 'SSD_sum': scalar(SSD_sum),
                'source_group_scale': scalar(source_group_scale), 'existing_group_scale': scalar(existing_group_scale),
                'serial_SSD': scalar(serial_SSD), 'scan':'z outer/y middle/x inner forward; masked FP32 terms promoted before each FP64 add'}
            def delta(reference, actual):
                r, a = scalar(reference), scalar(actual)
                return {'bitexact': r['FP64_little_endian_hex'] == a['FP64_little_endian_hex'],
                        'reference': r, 'actual': a, 'absolute_difference': float(abs(float(actual)-float(reference)))}
            report['own_scalar_comparison'] = {'serial_source_scale_vs_baseline': delta(baseline_scale, source_group_scale),
                'serial_existing_scale_vs_baseline': delta(baseline_scale, existing_group_scale),
                'serial_SSD_vs_baseline': delta(baseline_SSD, serial_SSD),
                'final_grouping_at_same_serial_sum': delta(existing_group_scale, source_group_scale)}
            report['own_math_completed_before_reference'] = True
            with paths['native_g'].open('rb') as stream:
                stream.seek(-8, os.SEEK_END)
                tail = stream.read(8)
            report['calls']['native_saved_g_tail8_read'] += 1
            require(len(tail) == 8, 'saved native scale element missing')
            native = struct.unpack('<d', tail)[0]
            report['posthoc_native_scale_comparison'] = {'native_scale': scalar(native),
                'baseline': delta(native, baseline_scale), 'serial_source_group': delta(native, source_group_scale),
                'serial_existing_group': delta(native, existing_group_scale),
                'input_cache_identity_unestablished': True,
                'interpretation':'Current-moving scalar-order contract; this is not the substituted-moving projection residual context.'}
            require(report['calls'] == {'NPZ_member_restore':4,'baseline_scalar_pair':1,'scalar_FP32_product_pair':1,
                                       'serial_reduction_pair':1,'native_saved_g_tail8_read':1}, 'finite call cap exceeded')
            report['control_completed'], report['status'], code = True, 'completed_not_integrated', 0
    except Exception as error:
        report['status'] = 'failed_first_guard_stopped_no_retry'
        report['exception'] = {'type':type(error).__name__,'message':str(error)}
        code = 124 if isinstance(error, TimeoutError) else 1
    finally:
        if torch is not None:
            report['flags_after'], report['FSL_DSO_after'] = flags(), mapped_fsl()
            report['flags_unchanged'] = report['flags_after'] == flags_before
            try:
                report['operands_after'] = {name:value_hash(value) for name,value in saved.items()}
                require(report['operands_after'] == report['operands_before'], 'operand changed')
                require(report['flags_unchanged'] and not report['FSL_DSO_after'], 'flags/CUDA/DSO postcondition changed')
            except Exception as error:
                report['postcheck_errors'].append({'type':type(error).__name__,'message':str(error)})
        if expected is not None:
            try:
                after, failed = check_bindings(args.root, expected)
                frozen, bad = check_freeze(args.workspace)
                report['bindings_after'], report['harness_after'] = after, frozen
                report['source_inputs_unchanged'] = after == before and frozen == freeze_before and not failed and not bad
                require(report['source_inputs_unchanged'], 'source/input/freeze changed')
                require(git_head(args.root/'repo') == args.canonical_main_commit, 'canonical commit changed')
                if header_before is not None:
                    report['header_after'] = fixed_header(file_paths(args.root, expected)['official_fixed'], expected['fixed_header'])
                    require(report['header_after'] == header_before, 'header changed')
            except Exception as error:
                report['postcheck_errors'].append({'type':type(error).__name__,'message':str(error)})
        if report['postcheck_errors']:
            code = code or 1
            report['control_completed'] = False
            report['status'] = 'failed_postcondition_stopped_no_retry'
        report['worker_until_summary_seconds'] = float(time.monotonic()-started)
        report['process_maxrss_KiB'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report['exit_code'] = int(code)
        write_json(args.output/'summary.private.json', report)
    print(json.dumps({'status':report['status'],'exit_code':code,'control_completed':report['control_completed']}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
