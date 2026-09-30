"""Train supervised multimodal components from BigFLICA-format NIfTI inputs."""

from __future__ import annotations

import contextlib
import csv
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from ..bigflica.pipeline import _device, _load_mask, _read_vector, _spatial_z, _write_maps
from .data import load_cohort, prepare_images
from .model import SupervisedComponents, SupervisedObjective


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')


def _batch(files: list[h5py.File], rows: np.ndarray, device: torch.device):
    batches = []
    for file in files:
        dataset = file['data']
        # Stores are chunked by subject; contiguous row reads avoid HDF5 fancy-index overhead.
        values = np.stack([dataset[int(row)] for row in rows])
        batches.append(torch.as_tensor(values, device=device, dtype=torch.float32))
    return batches


def _infer(model: SupervisedComponents, files: list[h5py.File], rows: np.ndarray,
           batch_size: int, device: torch.device):
    model.eval()
    latent, prediction = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            _, scores, estimates = model(_batch(files, rows[start:start + batch_size], device),
                                          reconstruct=False)
            latent.append(scores.cpu().numpy())
            prediction.append(estimates.cpu().numpy())
    return np.concatenate(latent), np.concatenate(prediction)


def _probabilities(values: np.ndarray) -> np.ndarray:
    exponentials = np.exp(values - values.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _validation_loss(prediction: np.ndarray, labels: np.ndarray,
                     targets: list[dict], sizes: list[int]) -> float:
    losses, offset = [], 0
    for i, (target, width) in enumerate(zip(targets, sizes)):
        observed = np.isfinite(labels[:, i])
        if not observed.any():
            raise ValueError(f"No validation labels for {target['name']}")
        head = prediction[observed, offset:offset + width]
        actual = labels[observed, i]
        if target['type'] == 'continuous':
            losses.append(float(np.mean((head[:, 0] - actual) ** 2)))
        else:
            # Stable log-softmax avoids underflow for confidently wrong labels.
            shifted = head - head.max(axis=1, keepdims=True)
            log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
            classes = actual.astype(int)
            weights = np.asarray(target.get('class_weights', [1.] * width), dtype=np.float64)[classes]
            losses.append(float(np.sum(-log_probability[np.arange(len(head)), classes] * weights)
                                / weights.sum()))
        offset += width
    return float(np.mean(losses))


def _metrics(prediction: np.ndarray, labels: np.ndarray, targets: list[dict],
             sizes: list[int]) -> dict:
    report, offset = {}, 0
    for i, (target, width) in enumerate(zip(targets, sizes)):
        observed = np.isfinite(labels[:, i])
        entry = {'n': int(observed.sum()), 'type': target['type']}
        head, actual = prediction[observed, offset:offset + width], labels[observed, i]
        if observed.any() and target['type'] == 'continuous':
            estimated = head[:, 0] * target['std'] + target['mean']
            actual = actual * target['std'] + target['mean']
            error = estimated - actual
            entry.update(mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error ** 2).mean())))
            variance = float(np.sum((actual - actual.mean()) ** 2))
            entry['r2'] = float(1 - np.sum(error ** 2) / variance) if variance > 0 else None
            entry['pearson_r'] = (float(np.corrcoef(estimated, actual)[0, 1])
                                  if len(actual) > 1 and np.std(estimated) > 0 and
                                  np.std(actual) > 0 else None)
        elif observed.any():
            probability, actual = _probabilities(head), actual.astype(int)
            estimated = probability.argmax(axis=1)
            entry.update(accuracy=float(np.mean(estimated == actual)),
                         balanced_accuracy=float(balanced_accuracy_score(actual, estimated)),
                         macro_f1=float(f1_score(actual, estimated, labels=range(width),
                                                 average='macro', zero_division=0)),
                         classes=target['classes'])
            if width == 2:
                entry['roc_auc'] = (float(roc_auc_score(actual, probability[:, 1]))
                                    if len(np.unique(actual)) == 2 else None)
        report[target['name']] = entry
        offset += width
    return report


def _prediction_rows(ids: list[str], splits: Sequence[str], prediction: np.ndarray,
                     labels: np.ndarray, targets: list[dict], sizes: list[int]):
    for row, subject_id in enumerate(ids):
        record = {'subject_id': subject_id, 'split': str(splits[row])}
        offset = 0
        for i, (target, width) in enumerate(zip(targets, sizes)):
            prefix, value = target['name'], labels[row, i]
            head = prediction[row, offset:offset + width]
            if target['type'] == 'continuous':
                record[f'{prefix}__prediction'] = float(head[0] * target['std'] + target['mean'])
                record[f'{prefix}__observed'] = (float(value * target['std'] + target['mean'])
                                               if np.isfinite(value) else '')
            else:
                probabilities = _probabilities(head[None])[0]
                record[f'{prefix}__prediction'] = target['classes'][int(probabilities.argmax())]
                record[f'{prefix}__observed'] = (target['classes'][int(value)]
                                               if np.isfinite(value) else '')
                for index, probability in enumerate(probabilities):
                    record[f'{prefix}__prob_{index:03d}'] = float(probability)
            offset += width
        yield record


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_superbigflica(subjects_root: str | Path,
                     modalities: Mapping[str, Mapping[str, str]],
                     phenotypes_csv: str | Path, targets: Mapping[str, str],
                     output_dir: str | Path, n_components: int, *,
                     id_column: str = 'subject_id', split_column: str | None = None,
                     subjects: Sequence[str] | None = None,
                     validation_fraction: float = 0.2, test_fraction: float = 0.2,
                     max_epochs: int = 50, batch_size: int = 64,
                     learning_rate: float = 0.001, dropout: float = 0.2,
                     relative_weight: float = 0.5, class_weight: str = 'balanced',
                     random_state: int = 0,
                     device: str = 'auto', max_gpu_gb: float = 19.0,
                     feature_block: int = 2048, top_voxels: int = 1000,
                     make_plots: bool = True) -> Path:
    """Fit and select on validation subjects; evaluate the held-out test once."""
    if (n_components < 1 or max_epochs < 1 or batch_size < 2 or
            not np.isfinite(learning_rate) or learning_rate <= 0 or
            not 0 <= dropout < 1 or not 0 < relative_weight < 1 or
            not np.isfinite(max_gpu_gb) or max_gpu_gb <= 0 or
            feature_block < 1 or top_voxels < 1):
        raise ValueError('Invalid component, optimization, block or memory parameters')
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use a new, empty output_dir to avoid overwriting a fitted model')
    start = time.perf_counter()
    cohort = load_cohort(subjects_root, modalities, phenotypes_csv, targets,
                         id_column=id_column, split_column=split_column,
                         validation_fraction=validation_fraction, test_fraction=test_fraction,
                         random_state=random_state, subjects=subjects,
                         class_weight=class_weight)
    training = np.flatnonzero(cohort.splits == 'train')
    validation = np.flatnonzero(cohort.splits == 'validation')
    if len(training) <= n_components + 1:
        raise ValueError('Training subjects must exceed n_components + 1 for spatial statistics')
    destination.mkdir(parents=True, exist_ok=True)
    specs = prepare_images(subjects_root, modalities, cohort, destination,
                           feature_block=feature_block)
    _write_json(destination / 'match_report.json', cohort.match_report)
    timings = {'match_load_normalize_s': time.perf_counter() - start}
    backend = _device(device)
    if backend.type == 'cuda':
        torch.cuda.synchronize(backend)
        torch.cuda.reset_peak_memory_stats(backend)
    n_features = [spec['n_features'] for spec in specs.values()]
    if n_components > sum(n_features):
        raise ValueError('n_components cannot exceed the total number of masked features')
    sizes = [1 if target['type'] == 'continuous' else len(target['classes'])
             for target in cohort.targets]
    budget = max_gpu_gb * 2 ** 30
    if backend.type == 'cuda':
        budget = min(budget, torch.cuda.mem_get_info(backend)[0] * 0.85)
    # Parameters, gradients, optimizer state, activations and backward buffers.
    parameter_bytes = 4 * 8 * (sum(n_features) * n_components +
                               (n_components + 1) * sum(sizes))
    per_subject_bytes = 4 * (5 * sum(n_features) + 16 * n_components + 4 * sum(sizes))
    effective_batch = min(batch_size, len(training))
    if backend.type == 'cuda':
        effective_batch = min(effective_batch, int((budget - parameter_bytes) / per_subject_bytes) - 1)
        if effective_batch < 2:
            raise ValueError('Spatial model and a two-subject batch exceed GPU budget; '
                             'reduce n_components or mask size')
    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)
    model = SupervisedComponents(n_features, n_components, sizes, dropout).to(backend)
    objective = SupervisedObjective(len(specs), cohort.targets, sizes, relative_weight).to(backend)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9)
    scale_optimizer = torch.optim.Adam(objective.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs)
    scale_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(scale_optimizer, max(1, max_epochs - 10))
    history, best_score, best_state, best_epoch = [], float('inf'), None, None
    training_start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(destination / 'input_store' / f'{name}.h5', 'r'))
                 for name in specs]
        for epoch in range(max_epochs):
            model.train()
            batches = list(np.array_split(rng.permutation(training),
                                          int(np.ceil(len(training) / effective_batch))))
            if len(batches[-1]) == 1:
                last = batches.pop()
                batches[-1] = np.concatenate([batches[-1], last])
            average_loss, average_terms = 0.0, np.zeros(4)
            for rows in batches:
                images = _batch(files, rows, backend)
                labels = torch.as_tensor(cohort.y[rows], device=backend)
                optimizer.zero_grad(set_to_none=True)
                scale_optimizer.zero_grad(set_to_none=True)
                rebuilt, _, prediction = model(images)
                loss, terms = objective(model, images, rebuilt, prediction, labels, len(training))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss; lower learning_rate')
                loss.backward()
                optimizer.step()
                if epoch >= 10:
                    scale_optimizer.step()
                fraction = len(rows) / len(training)
                average_loss += float(loss.detach()) * fraction
                average_terms += terms.detach().cpu().numpy() * fraction
            _, validation_prediction = _infer(model, files, validation, effective_batch, backend)
            score = _validation_loss(validation_prediction, cohort.y[validation], cohort.targets, sizes)
            if not np.isfinite(score):
                raise RuntimeError('Nonfinite validation loss')
            if score < best_score:
                best_score, best_epoch = score, epoch + 1
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_scales = {name: value.detach().cpu().clone() for name, value in objective.state_dict().items()}
            history.append(dict(epoch=epoch + 1, training_loss=average_loss,
                                validation_loss=score, reconstruction_loss=float(average_terms[0]),
                                spatial_l1=float(average_terms[1]), supervision_loss=float(average_terms[2]),
                                prediction_regularization=float(average_terms[3])))
            _write_csv(destination / 'history.csv', history)
            schedule.step()
            if epoch >= 10:
                scale_schedule.step()
        model.load_state_dict(best_state)
        latent, prediction = _infer(model, files, np.arange(len(cohort.ids)), effective_batch, backend)
        if not np.isfinite(latent).all() or not np.isfinite(prediction).all():
            raise RuntimeError('Selected model has nonfinite outputs')
        if backend.type == 'cuda':
            torch.cuda.synchronize(backend)
        timings['training_selection_prediction_s'] = time.perf_counter() - training_start
        _write_csv(destination / 'history.csv', history)
        _write_csv(destination / 'predictions.csv', list(_prediction_rows(
            cohort.ids, cohort.splits, prediction, cohort.y, cohort.targets, sizes)))
        np.save(destination / 'subj_course.npy', latent)
        with (destination / 'subj_course.tsv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['subject_id', 'split'] + [f'component_{i + 1:03d}' for i in range(n_components)])
            writer.writerows([[subject, split, *values] for subject, split, values in
                              zip(cohort.ids, cohort.splits, latent)])
        report = {}
        for split in ('train', 'validation', 'test'):
            rows = cohort.splits == split
            report[split] = _metrics(prediction[rows], cohort.y[rows], cohort.targets, sizes)
        _write_json(destination / 'metrics.json', report)
        spatial_start = time.perf_counter()
        design = np.column_stack([latent[training], np.ones(len(training), dtype=np.float32)]).astype(np.float32)
        if np.linalg.matrix_rank(design) < n_components + 1:
            raise ValueError('Selected component courses are rank deficient; reduce n_components')
        spatial_feature_block = feature_block
        statistic_block = max(feature_block, 1_000_000 // n_components)
        regression = None
        if backend.type == 'cuda':
            from ..bigflica.stats_torch import SpatialRegression, t_to_z_gpu
            # Training states and their last autograd graph are not needed for maps.
            del optimizer, scale_optimizer, schedule, scale_schedule, objective, model
            del images, labels, rebuilt, loss, terms
            torch.cuda.empty_cache()
            regression = SpatialRegression(latent[training].astype(np.float64), device=str(backend))
            available = min(max_gpu_gb * 2 ** 30 - torch.cuda.memory_allocated(backend),
                            torch.cuda.mem_get_info(backend)[0] * 0.85)
            spatial_feature_block = min(feature_block, int(available / (8 * 6 * len(training))))
            statistic_block = min(statistic_block, int(available / (8 * 64 * n_components)))
            if min(spatial_feature_block, statistic_block) < 1:
                raise ValueError('Spatial statistics exceed GPU budget; reduce n_components')
        for name, file in zip(specs, files):
            z = np.empty((file['data'].shape[1], n_components), dtype=np.float32)
            t_values = np.empty(z.shape, dtype=np.float64) if regression is not None else None
            for first in range(0, z.shape[0], spatial_feature_block):
                last = min(first + spatial_feature_block, z.shape[0])
                standardized = file['data'][training, first:last].T.astype(np.float64)
                if regression is None:
                    z[first:last] = _spatial_z(latent[training].astype(np.float64), standardized)
                else:
                    t_values[first:last] = regression.t(standardized)
            if regression is not None:
                for first in range(0, z.shape[0], statistic_block):
                    last = min(first + statistic_block, z.shape[0])
                    z[first:last] = t_to_z_gpu(t_values[first:last], regression.df, device=str(backend))
            np.save(destination / f'{name}_zstat.npy', z)
            _write_maps(name, z, *_load_mask(destination / specs[name]['mask']),
                        destination / 'maps', top_voxels)
        timings['spatial_maps_s'] = time.perf_counter() - spatial_start
    torch.save({'model': best_state, 'objective': best_scales}, destination / 'model.pt')
    np.save(destination / 'modality_weights.npy', best_state['modality_logits'].softmax(dim=1).numpy())
    np.save(destination / 'prediction_weights.npy', best_state['prediction_weight'].numpy())
    metadata = dict(schema_version=1, method='superbigflica', modalities=specs,
                    targets=cohort.targets, n_components=n_components, output_sizes=sizes,
                    dropout=dropout, relative_weight=relative_weight, class_weight=class_weight,
                    initialization='random', learning_rate=learning_rate,
                    max_epochs=max_epochs, batch_size=batch_size, effective_batch_size=effective_batch,
                    random_state=random_state, id_column=id_column, split_column=split_column,
                    validation_fraction=validation_fraction, test_fraction=test_fraction,
                    best_epoch=best_epoch, best_validation_loss=best_score,
                    model_selection='mean validation standardized MSE / training-weighted cross entropy',
                    counts={split: int(np.sum(cohort.splits == split))
                            for split in ('train', 'validation', 'test')},
                    device=str(backend), dtype='float32', spatial_statistics_dtype='float64', tf32=True,
                    max_gpu_gb=max_gpu_gb, feature_block=feature_block,
                    spatial_feature_block=spatial_feature_block, statistic_block=statistic_block,
                    top_voxels=top_voxels, make_plots=make_plots,
                    timings=timings, wall_time_s=time.perf_counter() - start,
                    peak_gpu_allocated_gib=(torch.cuda.max_memory_allocated(backend) / 2 ** 30
                                            if backend.type == 'cuda' else None),
                    upstream_commit='6695b638802aab50f43f889e97af223f8191018e',
                    reference='Gong et al. TMI 42(3):834-849; DOI:10.1109/TMI.2022.3218720')
    _write_json(destination / 'model.json', metadata)
    if make_plots:
        from .plotting import plot_superbigflica
        plot_start = time.perf_counter()
        plot_superbigflica(destination)
        timings['summary_plots_s'] = time.perf_counter() - plot_start
        metadata['wall_time_s'] = time.perf_counter() - start
        _write_json(destination / 'model.json', metadata)
    return destination


def apply_model(model_dir: str | Path, subject_dir: str | Path, *,
                device: str = 'auto', output_file: str | Path | None = None) -> dict:
    """Predict an unseen subject using frozen training masks, scales and model."""
    directory, subject = Path(model_dir), Path(subject_dir)
    metadata = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    if metadata.get('method') != 'superbigflica' or not subject.is_dir():
        raise ValueError('A SuperBigFLICA model and an existing subject_dir are required')
    with (directory / 'subj_course.tsv').open(encoding='utf-8') as stream:
        training_ids = {row['subject_id'] for row in csv.DictReader(stream, delimiter='\t')
                        if row['split'] == 'train'}
    if subject.name in training_ids:
        raise ValueError('apply_model requires a subject absent from training')
    backend = _device(device)
    model = SupervisedComponents([spec['n_features'] for spec in metadata['modalities'].values()],
                                  metadata['n_components'], metadata['output_sizes'],
                                  metadata['dropout']).to(backend)
    model.load_state_dict(torch.load(directory / 'model.pt', map_location=backend,
                                     weights_only=True)['model'])
    images = []
    for name, spec in metadata['modalities'].items():
        vector = _read_vector(subject / spec['image'], *_load_mask(directory / spec['mask']))
        normalized = (vector - np.load(directory / f'{name}_mean.npy')) / np.load(directory / f'{name}_std.npy')
        images.append(torch.as_tensor(normalized[None], device=backend, dtype=torch.float32))
    model.eval()
    with torch.inference_mode():
        _, latent, prediction = model(images, reconstruct=False)
    latent, prediction = latent.cpu().numpy()[0], prediction.cpu().numpy()[0]
    if not np.isfinite(latent).all() or not np.isfinite(prediction).all():
        raise RuntimeError('Nonfinite new-subject prediction')
    result = {'subject_id': subject.name, 'components': latent.tolist(), 'predictions': {}}
    offset = 0
    for target, width in zip(metadata['targets'], metadata['output_sizes']):
        head = prediction[offset:offset + width]
        if target['type'] == 'continuous':
            result['predictions'][target['name']] = {'value': float(head[0] * target['std'] + target['mean'])}
        else:
            probability = _probabilities(head[None])[0]
            result['predictions'][target['name']] = {
                'label': target['classes'][int(probability.argmax())],
                'probabilities': dict(zip(target['classes'], map(float, probability)))}
        offset += width
    if output_file is not None:
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, result)
    return result
