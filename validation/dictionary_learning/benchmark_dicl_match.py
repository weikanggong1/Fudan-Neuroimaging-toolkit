"""真实 R500/D200 三模态 DicL 的 CPU/GPU 一致性验证。

输入 projected：包含 vbm_projected.h5、fa_projected.h5、md_projected.h5 的目录。
每个文件的 data 为 float64、voxel×500，三阶段复用相同缓存，不运行 mMIGP/FLICA。
固定参数：D200、seed0、batch32、alpha1、最多1000 epochs、CPU8线程、GPU18GiB。
CPU 使用 sklearn 1.7.1；GPU 调用 FNIT 的生产函数。

位置参数（保持三阶段原有顺序）：
  cpu projected previous initial output
    previous：旧 stage_report.json 和 cpu_{vbm,fa,md}_dictionary.npy 所在目录；
              '-' 表示不请求旧输入 hash/最终字典核验，从零建立 CPU 参考。
    initial：旧 {vbm,fa,md}_sklearn_default_LU_initial_dictionary.npy 所在目录；
             '-' 表示不请求旧初始化核验。两项可独立省略。
  gpu projected cpu_reference output
    cpu_reference：已完成 cpu 阶段的输出目录，GPU 阶段不重跑 CPU。
  eval projected cpu_reference gpu_result output
    gpu_result：已完成 gpu 阶段的输出目录，eval 不重新训练。
  output：新建输出目录，必须不存在；report.json 记录 hash、步数、计时及验收指标。
          字典、随机排列、评估 voxel indices 和 codes 留在私密目录，不用于发布。

可选参数：
  --device：GPU 设备，默认 cuda:0。
  --events：GPU sparse_iterations，默认1000；实际早停步数另行记录。
  --voxels：eval 共同未访问训练批次中的 voxel 数，默认2000。
  --common-cpu-initialization：gpu 用冻结 CPU 初始原子；先执行原 GPU SVD 推进
                             RNG，再替换返回值，保持原 tensor stride 并记录两份 hash。
  --all-compatible-lars：gpu 每批使用兼容 LARS，作为因果诊断，不改变生产默认值。
  eval 中两个控制标记自动继承 GPU 报告；显式传入的标记必须与报告一致。
  --fixed-evaluation：仅 eval，复用既有 eval 输出目录中的 samples/indices；核对当前
                      CPU/GPU 批次均未访问这些 indices，以及冻结 CPU 规范化完全一致。
评估 voxel 仍参与全数据规范化/SVD，属于未访问训练批次诊断，不是 heldout 研究。

完整示例（替换中性绝对路径；PYTHONPATH 指向待核验 FNIT 源码）：
  PYTHONPATH=/absolute/FNIT/src python /absolute/FNIT/validation/dictionary_learning/benchmark_dicl_match.py cpu /absolute/projection - - /absolute/cpu_reference
  PYTHONPATH=/absolute/FNIT/src python /absolute/FNIT/validation/dictionary_learning/benchmark_dicl_match.py gpu /absolute/projection /absolute/cpu_reference /absolute/gpu_default --device cuda:0
  PYTHONPATH=/absolute/FNIT/src python /absolute/FNIT/validation/dictionary_learning/benchmark_dicl_match.py eval /absolute/projection /absolute/cpu_reference /absolute/gpu_default /absolute/evaluation_default
  PYTHONPATH=/absolute/FNIT/src python /absolute/FNIT/validation/dictionary_learning/benchmark_dicl_match.py gpu /absolute/projection /absolute/cpu_reference /absolute/gpu_common_initial --common-cpu-initialization
  PYTHONPATH=/absolute/FNIT/src python /absolute/FNIT/validation/dictionary_learning/benchmark_dicl_match.py eval /absolute/projection /absolute/cpu_reference /absolute/gpu_common_initial /absolute/evaluation_common_initial --fixed-evaluation /absolute/evaluation_default
原实现：sklearn MiniBatchDictionaryLearning / sparse_encode；
https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html
参考：Mairal et al., Online Learning for Matrix Factorization and Sparse Coding, JMLR 2010.
"""
import argparse
import contextlib
import copy
import gc
import hashlib
import json
import os
import pickle
import resource
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

NAMES = ('vbm', 'fa', 'md')
ATOMS, FEATURES, SEED, BATCH, MAX_EPOCHS = 200, 500, 0, 32, 1000
BASE_DRIVER_SHA256 = 'a14c750231828c53736ea58ec03f38c0c6a1415d190cd7a620dadff9cb3b6e71'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def emit(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def relative(a, b):
    norm = np.linalg.norm(a)
    return float(np.linalg.norm(a - b) / norm) if norm else (0.0 if not np.any(b) else None)


def numeric_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def clone_rng(rng):
    result = np.random.RandomState()
    result.set_state(rng.get_state())
    return result


def atom_match(cpu, gpu):
    denominator = np.linalg.norm(cpu, axis=1)[:, None] * np.linalg.norm(gpu, axis=1)[None, :]
    cosine = np.divide(cpu @ gpu.T, denominator, out=np.zeros_like(denominator), where=denominator != 0)
    rows, columns = linear_sum_assignment(-np.abs(cosine))
    assert np.array_equal(rows, np.arange(ATOMS))
    signs = np.where(cosine[rows, columns] < 0, -1., 1.)
    return columns, signs, {'direct_relative_L2': relative(cpu, gpu),
        'matched_relative_L2': relative(cpu, gpu[columns] * signs[:, None]),
        'matched_mean_cosine': float(np.abs(cosine[rows, columns]).mean()),
        'matched_min_cosine': float(np.abs(cosine[rows, columns]).min()),
        'matching_identity_permutation': bool(np.array_equal(columns, rows))}


def encoding_metrics(samples, dictionary, code, algorithm):
    residual = code @ dictionary - samples
    report = {'relative_residual': float(np.linalg.norm(residual) / np.linalg.norm(samples)),
              'SSE_per_voxel': float(np.square(residual).sum() / len(samples)),
              'nonzero_count': int(np.count_nonzero(code)),
              'max_nonzeros_per_voxel': int(np.count_nonzero(code, axis=1).max())}
    if algorithm == 'LASSO_alpha1':
        gradient = residual @ dictionary.T
        effective = np.abs(code) > 1e-8
        error = np.where(effective, np.abs(gradient + np.sign(code)), np.maximum(np.abs(gradient) - 1, 0))
        exact = np.where(code != 0, np.abs(gradient + np.sign(code)), np.maximum(np.abs(gradient) - 1, 0))
        report.update(objective_per_voxel=float((.5 * np.square(residual).sum() + np.abs(code).sum()) / len(samples)),
                      effective_support_threshold=1e-8, effective_KKT_max_absolute=float(error.max()),
                      exact_nonzero_KKT_max_absolute=float(exact.max()))
    return report


def main(args):
    import sklearn
    import torch
    from sklearn.decomposition import sparse_encode
    from threadpoolctl import threadpool_limits
    from fnit.dictionary_learning import cpu as dicl_cpu, torch_backend as dicl_torch
    if sklearn.__version__ != '1.7.1':
        raise RuntimeError('Requires the actual sklearn 1.7.1 reference environment')
    os.umask(0o077)
    torch.set_num_threads(8)
    backend = torch.device(args.device)
    projected, destination = Path(args.projected), Path(args.output)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    if args.phase == 'cpu':
        previous = None if args.previous == '-' else Path(args.previous)
        initial = None if args.initial == '-' else Path(args.initial)
        old = json.loads((previous / 'stage_report.json').read_text()) if previous else None
    else:
        cpu_directory = Path(args.cpu_reference)
        cpu_report = json.loads((cpu_directory / 'report.json').read_text())
        assert cpu_report['status'] == 'cpu_phase_complete'
        old = {'projection_sha256': cpu_report['projection_sha256']}
    report = {'status': 'preparing', 'modality_order': list(NAMES), 'R': FEATURES, 'D': ATOMS,
              'seed': SEED, 'batch_size': BATCH, 'max_epochs': MAX_EPOCHS,
              'alpha': 1., 'gpu_sparse_events': args.events, 'CPU_threads': 8,
              'sklearn_version': sklearn.__version__, 'driver_sha256': sha(__file__),
              'source_sha256': {'dictionary_learning.cpu': sha(dicl_cpu.__file__), 'dictionary_learning.torch_backend': sha(dicl_torch.__file__)},
              'projection_sha256': {}, 'CPU': {}, 'GPU': {}, 'comparisons': {},
              'phase': args.phase,
              'prespecified_acceptance': {'raw_matched_relative_L2_max': 1e-3,
                  'normalized_matched_relative_L2_max': 1e-3, 'matched_atom_cosine_min': .999999,
                  'LASSO_objective_relative_max': 1e-5, 'LASSO_reconstruction_relative_L2_max': 1e-3,
                  'OMP30_reconstruction_relative_L2_max': 1e-3, 'OMP30_code_equality_required': False},
              'scope': 'Same cached R500 projection; full DicL only. No MIGP, FLICA, model or maps.'}
    controls = {key: bool(getattr(args, key, None))
                for key in ('common_cpu_initialization', 'all_compatible_lars')}
    report['base_driver_sha256'] = BASE_DRIVER_SHA256
    report['causal_controls'] = controls
    report['causal_control_scope'] = 'Private opt-in GPU training interventions; unchanged CPU reference and acceptance gates'
    report['previous_projection_check'] = 'not_requested' if old is None else 'requested'
    report['CPU_reference_driver_sha256'] = report['driver_sha256'] if args.phase == 'cpu' else cpu_report['driver_sha256']
    report['GPU_training_driver_sha256'] = report['driver_sha256'] if args.phase == 'gpu' else None
    for name in NAMES:
        value = sha(projected / f'{name}_projected.h5')
        if old is not None:
            assert value == old['projection_sha256'][name]
        report['projection_sha256'][name] = value
    emit(destination / 'report.json', report)
    reference_models, raw, normalized, permutations = {}, {'cpu': {}, 'gpu': {}}, {'cpu': {}, 'gpu': {}}, {'cpu': {}, 'gpu': {}}
    original = dicl_cpu.MiniBatchDictionaryLearning
    current = {'name': None}
    if args.phase != 'cpu':
        report['CPU'] = copy.deepcopy(cpu_report['CPU'])
        report['CPU_reference_report_sha256'] = sha(cpu_directory / 'report.json')
        report['CPU_reference_source_sha256'] = cpu_report['source_sha256']
        for name in NAMES:
            raw['cpu'][name] = np.load(cpu_directory / f'cpu_{name}_raw.npy', allow_pickle=False)
            normalized['cpu'][name] = np.load(cpu_directory / f'cpu_{name}_normalized.npy', allow_pickle=False)
            permutations['cpu'][name] = np.load(cpu_directory / f'cpu_{name}_permutation.npy', allow_pickle=False)
            assert sha(cpu_directory / f'cpu_{name}_raw.npy') == report['CPU'][name]['raw_dictionary_sha256']
    if args.phase == 'eval':
        gpu_directory = Path(args.gpu_result)
        gpu_report = json.loads((gpu_directory / 'report.json').read_text())
        assert gpu_report['status'] == 'gpu_phase_complete'
        assert gpu_report['projection_sha256'] == report['projection_sha256']
        assert gpu_report['CPU_reference_report_sha256'] == report['CPU_reference_report_sha256']
        report['GPU'] = copy.deepcopy(gpu_report['GPU'])
        report['GPU_result_report_sha256'] = sha(gpu_directory / 'report.json')
        report['GPU_training_source_sha256'] = gpu_report['source_sha256']
        report['GPU_training_driver_sha256'] = gpu_report['driver_sha256']
        inherited = gpu_report.get('causal_controls', {key: False for key in controls})
        for key in controls:
            if getattr(args, key, None) is not None:
                assert bool(getattr(args, key)) == inherited[key], 'Evaluation flag disagrees with saved GPU training control'
        controls = copy.deepcopy(inherited)
        report['causal_controls'] = controls
        report['causal_controls_inherited_from_GPU_report'] = True
        if args.fixed_evaluation:
            fixed_directory = Path(args.fixed_evaluation)
            fixed_report = json.loads((fixed_directory / 'report.json').read_text())
            assert fixed_report['status'] == 'eval_phase_complete'
            assert fixed_report['projection_sha256'] == report['projection_sha256']
            assert fixed_report['CPU_reference_report_sha256'] == report['CPU_reference_report_sha256']
            report['fixed_evaluation_reference_report_sha256'] = sha(fixed_directory / 'report.json')
        for name in NAMES:
            raw['gpu'][name] = np.load(gpu_directory / f'gpu_{name}_raw.npy', allow_pickle=False)
            normalized['gpu'][name] = np.load(gpu_directory / f'gpu_{name}_normalized.npy', allow_pickle=False)
            permutations['gpu'][name] = np.load(gpu_directory / f'gpu_{name}_permutation.npy', allow_pickle=False)
            assert sha(gpu_directory / f'gpu_{name}_raw.npy') == report['GPU'][name]['raw_dictionary_sha256']
            model = original(**report['CPU'][name]['transform_model_parameters'])
            model.n_features_in_ = FEATURES
            model.components_ = raw['cpu'][name].copy()
            reference_models[name] = model

    class ObservedCPU(original):
        def _initialize_dict(self, samples, rng):
            dictionary = super()._initialize_dict(samples, rng)
            name = current['name']
            np.save(destination / f'cpu_{name}_initial.npy', dictionary.copy(order='K'))
            report['CPU'][name]['initial_reference_check'] = 'not_requested' if initial is None else 'requested'
            if initial is not None:
                expected = np.load(initial / f'{name}_sklearn_default_LU_initial_dictionary.npy', allow_pickle=False)
                report['CPU'][name]['initial_reference_relative_error'] = relative(expected, dictionary)
                report['CPU'][name]['initial_source_sha256'] = sha(initial / f'{name}_sklearn_default_LU_initial_dictionary.npy')
            report['CPU'][name]['initial_saved_sha256'] = sha(destination / f'cpu_{name}_initial.npy')
            report['CPU'][name]['post_initialization_RNG_protocol4_sha256'] = hashlib.sha256(pickle.dumps(rng.get_state(), protocol=4)).hexdigest()
            permutations['cpu'][name] = clone_rng(rng).permutation(len(samples))
            return dictionary

        def fit(self, samples, y=None):
            began = time.perf_counter()
            result = super().fit(samples, y)
            elapsed = time.perf_counter() - began
            name = current['name']
            # dicl_cpu.fit_dicl subsequently mutates the components_.T view.
            raw['cpu'][name] = self.components_.copy()
            reference_models[name] = self
            report['CPU'][name]['transform_model_parameters'] = self.get_params(deep=False)
            report['CPU'][name].update(learner_fit_seconds=elapsed, n_steps=int(self.n_steps_),
                n_iter=int(self.n_iter_), stopped_before_max_budget=bool(self.n_steps_ < MAX_EPOCHS * np.ceil(len(samples) / BATCH)))
            return result

    with threadpool_limits(limits=8):
        if args.phase == 'cpu':
            dicl_cpu.MiniBatchDictionaryLearning = ObservedCPU
            try:
                for name in NAMES:
                    current['name'] = name
                    report['status'] = f'CPU_fullfit_{name}'
                    report['CPU'][name] = {}
                    emit(destination / 'report.json', report)
                    with h5py.File(projected / f'{name}_projected.h5', 'r') as handle:
                        matrix = handle['data'][:]
                    assert matrix.shape[1] == FEATURES and matrix.dtype == np.float64
                    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
                    std[std == 0] = .1
                    np.savez(destination / f'{name}_CPU_normalization.npz', mean=mean, std=std)
                    began = time.perf_counter()
                    with (destination / f'CPU_{name}.log').open('w') as log, contextlib.redirect_stdout(log):
                        normalized['cpu'][name] = dicl_cpu.fit_dicl({name: matrix}, ATOMS, MAX_EPOCHS, SEED)[name]
                    report['CPU'][name]['production_call_seconds'] = time.perf_counter() - began
                    reference_models[name].components_ = raw['cpu'][name].copy()
                    report['CPU'][name]['previous_normalized_check'] = 'not_requested' if previous is None else 'requested'
                    if previous is not None:
                        prior = np.load(previous / f'cpu_{name}_dictionary.npy', allow_pickle=False)
                        equal = bool(np.array_equal(prior, normalized['cpu'][name]))
                        report['CPU'][name].update(previous_normalized_array_exact_equal=equal,
                            previous_normalized_relative_error=relative(prior, normalized['cpu'][name]),
                            previous_normalized_file_sha256=sha(previous / f'cpu_{name}_dictionary.npy'))
                    np.save(destination / f'cpu_{name}_raw.npy', raw['cpu'][name])
                    np.save(destination / f'cpu_{name}_normalized.npy', normalized['cpu'][name])
                    np.save(destination / f'cpu_{name}_permutation.npy', permutations['cpu'][name])
                    report['CPU'][name]['raw_dictionary_sha256'] = sha(destination / f'cpu_{name}_raw.npy')
                    emit(destination / 'report.json', report)
                    if previous is not None:
                        assert equal, 'CPU reference numerical replay changed; inspect report before GPU fitting'
                    del matrix
                    gc.collect()
            finally:
                dicl_cpu.MiniBatchDictionaryLearning = original

        if args.phase == 'gpu':
            original_init, original_updater = dicl_torch._randomized_svd_dictionary, dicl_torch._DictionaryUpdater
            original_solver = dicl_torch._SparseCodesBPDN
            solver_observations = []
            last_raw = {}

            def observed_init(dataset, samples, mean, std, atoms, rng, block):
                name = Path(dataset.file.filename).name.removesuffix('_projected.h5')
                current['name'] = name
                dictionary = original_init(dataset, samples, mean, std, atoms, rng, block)
                values = dictionary.cpu().numpy()
                cpu_initial_path = cpu_directory / f'cpu_{name}_initial.npy'
                assert sha(cpu_initial_path) == report['CPU'][name]['initial_saved_sha256']
                cpu_initial = np.load(cpu_initial_path, allow_pickle=False)
                assert cpu_initial.shape == values.shape == (ATOMS, FEATURES)
                assert cpu_initial.dtype == values.dtype == np.float64 and np.isfinite(cpu_initial).all()
                np.save(destination / f'gpu_{name}_initial.npy', values)
                np.savez(destination / f'{name}_GPU_normalization.npz', mean=mean.cpu().numpy(), std=std.cpu().numpy())
                permutations['gpu'][name] = clone_rng(rng).permutation(dataset.shape[0])
                report['GPU'][name] = {'n_steps': 0, 'initial_saved_sha256': sha(destination / f'gpu_{name}_initial.npy'),
                    'initial_CPU_relative_error': relative(cpu_initial, values),
                    'post_initialization_RNG_protocol4_sha256': hashlib.sha256(pickle.dumps(rng.get_state(), protocol=4)).hexdigest()}
                if controls['common_cpu_initialization']:
                    assert report['GPU'][name]['post_initialization_RNG_protocol4_sha256'] == report['CPU'][name]['post_initialization_RNG_protocol4_sha256'], 'GPU and CPU post-initialization RNG states differ'
                    returned = torch.empty_like(dictionary)
                    returned.copy_(torch.as_tensor(cpu_initial, device=dictionary.device, dtype=dictionary.dtype))
                else:
                    returned = dictionary
                assert returned.stride() == dictionary.stride(), 'Replacement changed the production dictionary layout'
                returned_values = returned.cpu().numpy()
                np.save(destination / f'gpu_{name}_returned_initial.npy', returned_values)
                report['GPU'][name].update(common_cpu_initialization=controls['common_cpu_initialization'],
                    initialization_label=('CPU_reference_atoms_after_executed_original_GPU_SVD'
                        if controls['common_cpu_initialization'] else 'Original_production_GPU_SVD_atoms'),
                    original_GPU_initial_numeric_sha256=numeric_sha(values),
                    CPU_initial_numeric_sha256=numeric_sha(cpu_initial),
                    actual_returned_initial_sha256=sha(destination / f'gpu_{name}_returned_initial.npy'),
                    actual_returned_initial_numeric_sha256=numeric_sha(returned_values),
                    actual_returned_initial_CPU_relative_error=relative(cpu_initial, returned_values),
                    actual_returned_initial_CPU_array_exact_equal=bool(np.array_equal(cpu_initial, returned_values)),
                    original_GPU_initial_stride=list(dictionary.stride()),
                    actual_returned_initial_stride=list(returned.stride()),
                    actual_returned_initial_stride_equal=True,
                    GPU_CPU_post_initialization_RNG_equal=(report['GPU'][name]['post_initialization_RNG_protocol4_sha256'] == report['CPU'][name]['post_initialization_RNG_protocol4_sha256']),
                    post_returned_initialization_RNG_protocol4_sha256=hashlib.sha256(pickle.dumps(rng.get_state(), protocol=4)).hexdigest(),
                    original_GPU_SVD_executed=True)
                assert report['GPU'][name]['post_initialization_RNG_protocol4_sha256'] == report['GPU'][name]['post_returned_initialization_RNG_protocol4_sha256']
                if controls['common_cpu_initialization']:
                    assert report['GPU'][name]['actual_returned_initial_CPU_array_exact_equal']
                return returned

            class ObservedUpdater(original_updater):
                def __call__(self, dictionary, a, b, samples, rng):
                    result = super().__call__(dictionary, a, b, samples, rng)
                    name = current['name']
                    report['GPU'][name]['n_steps'] += 1
                    # Independent clone survives production's final in-place centering.
                    last_raw[name] = dictionary.clone()
                    return result

            class ObservedSolver(original_solver):
                def __init__(self, *positional, **keyword):
                    if controls['all_compatible_lars']:
                        keyword['compatibility_mode'] = True
                    super().__init__(*positional, **keyword)

                def __call__(self, *positional, **keyword):
                    counters = ('calls', 'fallback_count', 'polish_checks', 'near_node_fallback_count')
                    before = {key: int(getattr(self, key, 0)) for key in counters}
                    try:
                        return super().__call__(*positional, **keyword)
                    finally:
                        solver_observations.append({'modality': current['name'],
                            'compatibility_mode': bool(getattr(self, 'compatibility_mode', False)),
                            'declared_solver_mode': str(getattr(self, 'solver_mode', 'production_BPDN')),
                            **{key: int(getattr(self, key, 0)) - before[key] for key in counters}})

            torch.cuda.set_device(backend)
            torch.cuda.init()
            fraction = min(1., 18 * 2**30 / torch.cuda.get_device_properties(backend).total_memory)
            torch.cuda.set_per_process_memory_fraction(fraction, backend)
            report['allocator_hardlimit_gib'] = 18
            report['allocator_fraction'] = fraction
            torch.cuda.reset_peak_memory_stats(backend)
            report['status'] = 'GPU_fullfit_all_three_modalities'
            emit(destination / 'report.json', report)
            dicl_torch._randomized_svd_dictionary, dicl_torch._DictionaryUpdater = observed_init, ObservedUpdater
            dicl_torch._SparseCodesBPDN = ObservedSolver
            began = time.perf_counter()
            try:
                with (destination / 'GPU_all_modalities.log').open('w') as log, contextlib.redirect_stdout(log):
                    normalized['gpu'] = dicl_torch.fit_dicl_gpu_streaming(projected, NAMES, ATOMS,
                        device=args.device, max_iter=MAX_EPOCHS, batch_size=BATCH,
                        sparse_iterations=args.events, alpha=1., random_state=SEED, feature_block=2048)
                torch.cuda.synchronize(backend)
            finally:
                dicl_torch._randomized_svd_dictionary, dicl_torch._DictionaryUpdater = original_init, original_updater
                dicl_torch._SparseCodesBPDN = original_solver
            report['GPU_production_all_modalities_seconds'] = time.perf_counter() - began
            report['GPU_timing_note'] = 'Includes initialization, input I/O and observer snapshot clones; not an uninstrumented speed benchmark'
            report['peak_GPU_allocated_gib'] = torch.cuda.max_memory_allocated(backend) / 2**30
            report['GPU_budget_18gib_pass'] = report['peak_GPU_allocated_gib'] <= 18
            for name in NAMES:
                raw['gpu'][name] = last_raw[name].cpu().numpy()
                np.save(destination / f'gpu_{name}_raw.npy', raw['gpu'][name])
                np.save(destination / f'gpu_{name}_normalized.npy', normalized['gpu'][name])
                np.save(destination / f'gpu_{name}_permutation.npy', permutations['gpu'][name])
                report['GPU'][name]['raw_dictionary_sha256'] = sha(destination / f'gpu_{name}_raw.npy')
                batches_per_epoch = int(np.ceil(len(permutations['gpu'][name]) / BATCH))
                report['GPU'][name]['n_iter'] = int(np.ceil(report['GPU'][name]['n_steps'] / batches_per_epoch))
                report['GPU'][name]['stopped_before_max_budget'] = report['GPU'][name]['n_steps'] < MAX_EPOCHS * batches_per_epoch
                relevant = [value for value in solver_observations if value['modality'] == name]
                stats = {key: sum(value[key] for value in relevant)
                         for key in ('calls', 'fallback_count', 'polish_checks', 'near_node_fallback_count')}
                stats['solver_mode'] = ('All_compatible_LARS_diagnostic' if controls['all_compatible_lars']
                    else ('ADMM_polish_with_guarded_fallbacks' if stats['polish_checks'] else 'LARS_fallback_only_observed'))
                stats['compatibility_modes_observed'] = sorted({value['compatibility_mode'] for value in relevant})
                if controls['all_compatible_lars']:
                    assert stats['compatibility_modes_observed'] == [True]
                stats['declared_modes'] = sorted({value['declared_solver_mode'] for value in relevant})
                stats['mode_interpretation'] = 'Derived from observed polishing; observers never modify solver counters'
                report['GPU'][name]['solver_stats'] = stats
            last_raw.clear()
            torch.cuda.empty_cache()

        if args.phase == 'eval':
            for name in NAMES:
                report['status'] = f'CPU_reference_encoding_evaluation_{name}'
                emit(destination / 'report.json', report)
                columns, signs, matched_raw = atom_match(raw['cpu'][name], raw['gpu'][name])
                _, _, matched_normalized = atom_match(normalized['cpu'][name], normalized['gpu'][name])
                n_voxels = len(permutations['cpu'][name])
                seen = np.zeros(n_voxels, dtype=bool)
                for engine in ('cpu', 'gpu'):
                    visited = min(n_voxels, report[engine.upper()][name]['n_steps'] * BATCH)
                    seen[permutations[engine][name][:visited]] = True
                unseen = np.flatnonzero(~seen)
                if not args.fixed_evaluation and len(unseen) < args.voxels:
                    report['comparisons'][name] = {'raw': matched_raw, 'normalized': matched_normalized,
                        'evaluation_status': 'Insufficient jointly unseen training-batch voxels; no heldout substitution'}
                    continue
                fixed_sha = None
                if args.fixed_evaluation:
                    fixed_path = fixed_directory / f'private_{name}_evaluation_samples.npz'
                    fixed_sha = sha(fixed_path)
                    with np.load(fixed_path, allow_pickle=False) as frozen:
                        indices, frozen_samples = frozen['voxel_indices'].copy(), frozen['samples'].copy()
                    assert indices.ndim == 1 and len(indices) and np.issubdtype(indices.dtype, np.integer)
                    assert np.all(np.diff(indices) > 0) and indices[0] >= 0 and indices[-1] < n_voxels
                    assert not seen[indices].any(), 'Fixed evaluation includes voxels visited by current training batches'
                    assert frozen_samples.shape == (len(indices), FEATURES) and np.isfinite(frozen_samples).all()
                else:
                    indices = np.sort(np.random.RandomState(1701).choice(unseen, args.voxels, replace=False))
                with h5py.File(projected / f'{name}_projected.h5', 'r') as handle:
                    samples = handle['data'][indices]
                with np.load(cpu_directory / f'{name}_CPU_normalization.npz') as stats:
                    samples = (samples - stats['mean']) / stats['std']
                if args.fixed_evaluation:
                    assert np.array_equal(samples, frozen_samples), 'Fixed samples disagree with current input and frozen CPU normalization'
                    samples = frozen_samples
                np.savez(destination / f'private_{name}_evaluation_samples.npz', samples=samples, voxel_indices=indices)
                comparison = {'raw': matched_raw, 'normalized': matched_normalized,
                    'scope': 'Jointly unseen by actual CPU/GPU training batches; used in full-data normalization/SVD, not heldout voxels or subjects',
                    'jointly_unseen_training_batch_voxels': len(unseen), 'evaluation_voxels': len(indices),
                    'evaluation_samples_sha256': sha(destination / f'private_{name}_evaluation_samples.npz'),
                    'evaluation_samples_numeric_sha256': numeric_sha(samples),
                    'evaluation_indices_numeric_sha256': numeric_sha(indices),
                    'fixed_evaluation_reused': bool(args.fixed_evaluation),
                    'fixed_evaluation_source_samples_sha256': fixed_sha,
                    'fixed_evaluation_current_batches_unseen_verified': bool(args.fixed_evaluation),
                    'fixed_evaluation_CPU_normalization_exact_verified': bool(args.fixed_evaluation),
                    'same_common_normalizer': 'Frozen actual CPU axis0 mean/std', 'encoding': {},
                    'gates': {'raw_dictionary_L2': matched_raw['matched_relative_L2'] <= 1e-3,
                              'normalized_dictionary_L2': matched_normalized['matched_relative_L2'] <= 1e-3,
                              'raw_atom_cosine': matched_raw['matched_min_cosine'] >= .999999,
                              'normalized_atom_cosine': matched_normalized['matched_min_cosine'] >= .999999}}
                for algorithm in ('LASSO_alpha1', 'OMP30_actual_transform'):
                    codes, metrics = {}, {}
                    began = time.perf_counter()
                    for engine in ('cpu', 'gpu'):
                        dictionary = raw[engine][name]
                        if algorithm == 'LASSO_alpha1':
                            code = sparse_encode(samples, dictionary, algorithm='lasso_lars', alpha=1., max_iter=1000, n_jobs=1)
                        else:
                            model = copy.deepcopy(reference_models[name])
                            model.components_ = dictionary.copy()
                            assert model.transform_algorithm == 'omp' and model.transform_n_nonzero_coefs == 30
                            code = model.transform(samples)
                        codes[engine] = code
                        metrics[engine] = encoding_metrics(samples, dictionary, code, algorithm)
                        np.save(destination / f'private_{name}_{algorithm}_{engine}_codes.npy', code)
                    comparison['encoding'][algorithm] = {'CPU': metrics['cpu'], 'GPU_dictionary_CPU_encoder': metrics['gpu'],
                        'matched_code_relative_L2': relative(codes['cpu'], codes['gpu'][:, columns] * signs),
                        'reconstruction_relative_L2': relative(codes['cpu'] @ raw['cpu'][name], codes['gpu'] @ raw['gpu'][name]),
                        'evaluation_seconds': time.perf_counter() - began}
                    reconstruction_error = comparison['encoding'][algorithm]['reconstruction_relative_L2']
                    comparison['gates'][algorithm + '_reconstruction'] = reconstruction_error is not None and reconstruction_error <= 1e-3
                    if algorithm == 'LASSO_alpha1':
                        objective_error = abs(metrics['cpu']['objective_per_voxel'] - metrics['gpu']['objective_per_voxel']) / metrics['cpu']['objective_per_voxel']
                        comparison['encoding'][algorithm]['objective_relative_error'] = objective_error
                        comparison['gates']['LASSO_objective'] = objective_error <= 1e-5
                comparison['all_prespecified_gates_pass'] = all(comparison['gates'].values())
                report['comparisons'][name] = comparison
                emit(destination / 'report.json', report)
    if args.phase == 'eval':
        report['all_modalities_prespecified_gates_pass'] = all(
            report['comparisons'][name].get('all_prespecified_gates_pass', False) for name in NAMES)
    report['status'] = args.phase + '_phase_complete'
    report['process_peak_RSS_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    for path in destination.iterdir():
        if path.is_file():
            path.chmod(0o600)
    emit(destination / 'report.json', report)
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='phase', required=True)
    for phase, arguments in (
        ('cpu', ('projected', 'previous', 'initial', 'output')),
        ('gpu', ('projected', 'cpu_reference', 'output')),
        ('eval', ('projected', 'cpu_reference', 'gpu_result', 'output')),
    ):
        child = subparsers.add_parser(phase)
        for name in arguments:
            child.add_argument(name)
        child.add_argument('--device', default='cuda:0')
        child.add_argument('--events', type=int, default=1000)
        child.add_argument('--voxels', type=int, default=2000)
        if phase in ('gpu', 'eval'):
            child.add_argument('--common-cpu-initialization', action='store_true', default=None)
            child.add_argument('--all-compatible-lars', action='store_true', default=None)
        if phase == 'eval':
            child.add_argument('--fixed-evaluation', help='Reuse an existing eval directory samples/indices, validating current unseen batches and CPU normalization')
    main(parser.parse_args())
