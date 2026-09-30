"""Editable figures for fitted supervised components and held-out predictions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import nibabel as nib
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve


COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9')


def _ranking(courses: np.ndarray, observed: np.ndarray, kind: str, classes: int = 0):
    """Rank with training labels; a reversed binary direction is also a signal."""
    valid = np.isfinite(observed)
    scores = courses[valid]
    values = observed[valid]
    result = np.zeros(courses.shape[1])
    if kind == 'continuous':
        metric = 'absolute_training_pearson_r'
        if len(values) > 1 and np.std(values) > 0:
            for column in range(scores.shape[1]):
                if np.std(scores[:, column]) > 0:
                    result[column] = abs(np.corrcoef(scores[:, column], values)[0, 1])
    elif classes == 2:
        metric = 'training_auc_either_direction'
        if len(np.unique(values)) == 2:
            for column in range(scores.shape[1]):
                value = roc_auc_score(values, scores[:, column])
                result[column] = max(value, 1 - value)
    else:
        metric = 'training_multiclass_eta_squared'
        for column in range(scores.shape[1]):
            vector = scores[:, column]
            total = np.sum((vector - vector.mean()) ** 2)
            between = sum(np.sum(values == category) *
                          (vector[values == category].mean() - vector.mean()) ** 2
                          for category in np.unique(values))
            result[column] = between / total if total > 0 else 0
    order = np.argsort(-result, kind='stable')[:min(3, len(result))]
    return metric, result, order


def _save(figure, destination: Path):
    import matplotlib.pyplot as plt
    for extension in ('png', 'svg', 'pdf'):
        figure.savefig(destination.with_suffix('.' + extension), dpi=300,
                       bbox_inches='tight', facecolor='white')
    plt.close(figure)


def _axis_style(axis):
    axis.spines[['top', 'right']].set_visible(False)
    axis.tick_params(direction='out', length=2.5, width=.6, pad=2)
    axis.grid(False)


def _brain_figure(directory: Path, metadata: dict, indices: np.ndarray,
                  values: np.ndarray, metric: str, name: str, destination: Path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    modalities = list(metadata['modalities'])
    figure = plt.figure(figsize=(7.2, 1.35 * len(indices) + .5))
    grid = figure.add_gridspec(len(indices), len(modalities), hspace=.35, wspace=.08)
    cmap = LinearSegmentedColormap.from_list('signed_z', ['#0072B2', '#ffffff', '#D55E00'])
    backgrounds, images = {}, {}
    bound = 1.0
    for modality, specification in metadata['modalities'].items():
        mask_image = nib.load(str(directory / specification['mask']))
        mask = np.asarray(mask_image.dataobj) > 0
        mean = np.zeros(mask.shape, dtype=np.float32)
        mean[mask] = np.load(directory / f'{modality}_mean.npy')
        backgrounds[modality] = np.asarray(nib.as_closest_canonical(
            nib.Nifti1Image(mean, mask_image.affine)).dataobj)
        for component in indices:
            count = min(metadata['top_voxels'], specification['n_features'])
            path = directory / 'maps' / modality / f'component-{component + 1:03d}_top-{count}.nii.gz'
            image = nib.as_closest_canonical(nib.load(str(path)))
            images[modality, int(component)] = image
            bound = max(bound, float(np.max(np.abs(np.asarray(image.dataobj)))))
    for row, component in enumerate(indices):
        for column, modality in enumerate(modalities):
            subgrid = grid[row, column].subgridspec(1, 3, wspace=.025)
            image = images[modality, int(component)]
            data = np.asarray(image.dataobj)
            voxel_sizes = nib.affines.voxel_sizes(image.affine)
            background = backgrounds[modality]
            peak = np.asarray(np.unravel_index(np.abs(data).argmax(), data.shape))
            location = nib.affines.apply_affine(image.affine, peak)
            nonzero = background[background != 0]
            low, high = (np.percentile(nonzero, [2, 98]) if nonzero.size else (0, 1))
            if high <= low:
                high = low + 1
            for orientation in range(3):
                axis = figure.add_subplot(subgrid[0, orientation])
                plane_axes = [dimension for dimension in range(3) if dimension != orientation]
                aspect = voxel_sizes[plane_axes[1]] / voxel_sizes[plane_axes[0]]
                layer = np.rot90(np.take(data, peak[orientation], axis=orientation))
                gray = np.rot90(np.take(background, peak[orientation], axis=orientation))
                axis.imshow(np.ma.masked_where(gray == 0, gray), cmap='gray',
                            vmin=low, vmax=high, interpolation='nearest', aspect=aspect)
                shown = axis.imshow(np.ma.masked_where(layer == 0, layer), cmap=cmap,
                                    vmin=-bound, vmax=bound, interpolation='nearest', alpha=.9,
                                    aspect=aspect)
                axis.set_axis_off()
                axis.set_title(f'{"xyz"[orientation]}={location[orientation]:.0f} mm', fontsize=6, pad=2)
                if orientation == 1:
                    axis.text(.5, 1.3, f'{modality} · latent {component + 1:02d}',
                              transform=axis.transAxes, ha='center', fontsize=7)
                if column == 0 and orientation == 0:
                    axis.text(-.04, .5, f'{chr(97 + row)}  {values[component]:.3f}',
                              transform=axis.transAxes, rotation=90, va='center', ha='right', fontsize=7)
    figure.subplots_adjust(top=.84, bottom=.11, left=.06, right=.97)
    color_axis = figure.add_axes([.38, .025, .24, .015])
    figure.colorbar(shown, cax=color_axis, orientation='horizontal', label='Signed z')
    figure.suptitle(f'{name} · top training signals', fontsize=8, y=.99)
    figure.text(.5, .91, metric.replace('_', ' '), ha='center', fontsize=6)
    _save(figure, destination)


def _roc_figure(observed: np.ndarray, probabilities: np.ndarray, classes: list[str],
                name: str, destination: Path) -> dict:
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(3.54, 3.1))
    class_auc, grid, interpolated = {}, np.linspace(0, 1, 201), []
    result = {'class_auc': class_auc, 'macro_auc': None, 'micro_auc': None}
    for category in range(1 if len(classes) == 2 else 0, len(classes)):
        binary = observed == category
        if len(np.unique(binary)) < 2:
            class_auc[classes[category]] = None
            continue
        false_positive, true_positive, _ = roc_curve(binary, probabilities[:, category])
        score = auc(false_positive, true_positive)
        axis.plot(false_positive, true_positive, color=COLORS[category % len(COLORS)], lw=1,
                  label=f'{classes[category]}  AUC={score:.3f}')
        class_auc[classes[category]] = float(score)
        interpolated.append(np.interp(grid, false_positive, true_positive))
    if len(classes) > 2:
        complete = len(interpolated) == len(classes)
        result['macro_auc'] = (float(np.mean(list(class_auc.values()))) if complete else None)
        if complete:
            mean_curve = np.mean(interpolated, axis=0)
            axis.plot(grid, mean_curve, color='black', lw=1.2,
                      label=f'Macro AUC={result["macro_auc"]:.3f}')
        elif len(observed):
            axis.text(.04, .96, 'Macro AUC undefined: a test class is absent',
                      transform=axis.transAxes, va='top', fontsize=6)
        if len(observed):
            encoded = np.eye(len(classes))[observed.astype(int)]
            fpr, tpr, _ = roc_curve(encoded.ravel(), probabilities.ravel())
            axis.plot(fpr, tpr, color='#777777', ls='--', lw=.9,
                      label=f'Micro AUC={auc(fpr, tpr):.3f}')
            result['micro_auc'] = float(auc(fpr, tpr))
    axis.plot([0, 1], [0, 1], color='#777777', lw=.7, ls=':')
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False positive rate', ylabel='True positive rate')
    axis.set_title(f'{name} · test n={len(observed)}', fontsize=7)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(frameon=False, fontsize=6, loc='lower right')
    else:
        axis.text(.5, .5, 'ROC undefined: a test class is absent', transform=axis.transAxes,
                  ha='center', wrap=True, fontsize=7)
    _axis_style(axis)
    figure.tight_layout()
    _save(figure, destination)
    return result


def plot_superbigflica(model_dir: str | Path, output_dir: str | Path | None = None, *,
                       labels: Mapping[str, str] | None = None) -> Path:
    """Plot saved model weights, training-selected maps and held-out predictions.

    Saves 300-dpi PNG plus editable SVG/PDF. Component ranking uses training
    labels only; scatter and ROC use test subjects. No subject IDs are exported
    in the aggregate figure summary. Backgrounds are frozen modality train means.
    Optional labels map CSV target names to display names including units.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(model_dir)
    metadata = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    if metadata.get('method') != 'superbigflica':
        raise ValueError('A fitted SuperBigFLICA model is required')
    destination = Path(output_dir) if output_dir is not None else directory / 'plots'
    destination.mkdir(parents=True, exist_ok=True)
    with (directory / 'subj_course.tsv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    with (directory / 'predictions.csv').open(encoding='utf-8') as stream:
        predictions = {row['subject_id']: row for row in csv.DictReader(stream)}
    courses = np.load(directory / 'subj_course.npy')
    if courses.shape != (len(rows), metadata['n_components']):
        raise ValueError('Saved courses and subject table differ')
    aligned = [predictions[row['subject_id']] for row in rows]
    if any(row['split'] != predicted['split'] for row, predicted in zip(rows, aligned)):
        raise ValueError('Prediction and course splits differ')
    training = np.asarray([row['split'] == 'train' for row in rows])
    testing = np.asarray([row['split'] == 'test' for row in rows])
    weights = np.load(directory / 'prediction_weights.npy')
    summary = {'ranking_split': 'train', 'prediction_split': 'test', 'targets': {},
               'brain_background': 'frozen modality training mean',
               'brain_slices': 'orthogonal voxel planes through each map maximum absolute z',
               'style': 'Nature figure guide; 300-dpi PNG and editable SVG/PDF'}
    style = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
             'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'axes.linewidth': .6,
             'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
             'axes.grid': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white'}
    with plt.rc_context(style):
        figure, axes = plt.subplots(len(metadata['targets']), 1,
                                    figsize=(7.2, 2.1 * len(metadata['targets'])), squeeze=False)
        offset = 0
        for index, (target, width) in enumerate(zip(metadata['targets'], metadata['output_sizes'])):
            name, kind = target['name'], target['type']
            display_name = (labels or {}).get(name, name)
            classes = target.get('classes', [])
            encoding = {label: category for category, label in enumerate(classes)}
            observed = np.asarray([np.nan if not row[f'{name}__observed'] else
                                   float(row[f'{name}__observed']) if kind == 'continuous' else
                                   encoding[row[f'{name}__observed']] for row in aligned])
            metric, signal, selected = _ranking(courses[training], observed[training], kind, len(classes))
            entry = {'type': kind, 'label': display_name, 'ranking_metric': metric,
                     'training_signal_by_component': signal.tolist(),
                     'selected_components': (selected + 1).tolist()}
            head = weights[:, offset:offset + width]
            axis, positions = axes[index, 0], np.arange(metadata['n_components']) + 1
            if kind == 'continuous' or width == 2:
                coefficients = head[:, 0] if kind == 'continuous' else head[:, 1] - head[:, 0]
                bars = axis.bar(positions, coefficients, width=.7,
                                color=np.where(coefficients >= 0, COLORS[0], COLORS[1]),
                                edgecolor='black', linewidth=.3)
                for component in selected:
                    bars[component].set_linewidth(1.1)
                description = 'Standardized phenotype coefficient' if kind == 'continuous' else 'Log-odds coefficient (class 1 − class 0)'
                entry['plotted_coefficients'] = coefficients.tolist()
            else:
                centered = head - head.mean(axis=1, keepdims=True)
                for category, label in enumerate(classes):
                    axis.bar(positions + (category - (width - 1) / 2) * .8 / width,
                             centered[:, category], width=.8 / width,
                             color=COLORS[category % len(COLORS)], label=label)
                axis.legend(frameon=False, ncol=min(width, 4), loc='upper right')
                description = 'Mean-centered class-logit coefficient'
                entry['plotted_coefficients'] = centered.tolist()
            axis.axhline(0, color='black', lw=.6)
            axis.set(xlabel='Latent component', ylabel='Weight', xticks=positions)
            axis.set_title(f'{display_name} · {description}', loc='left')
            _axis_style(axis)
            _brain_figure(directory, metadata, selected, signal, metric, display_name,
                          destination / f'target-{index + 1:03d}_top-components')
            valid = testing & np.isfinite(observed)
            if kind == 'continuous':
                actual = observed[valid]
                estimated = np.asarray([float(row[f'{name}__prediction']) for row in aligned])[valid]
                scatter, scatter_axis = plt.subplots(figsize=(3.54, 3.1))
                scatter_axis.scatter(actual, estimated, s=8, alpha=.4, color=COLORS[0], linewidths=0, rasterized=True)
                if len(actual):
                    low, high = min(actual.min(), estimated.min()), max(actual.max(), estimated.max())
                    scatter_axis.plot([low, high], [low, high], color='black', lw=.7, ls='--')
                    error = estimated - actual
                    correlation = (float(np.corrcoef(actual, estimated)[0, 1]) if len(actual) > 1
                                   and np.std(actual) > 0 and np.std(estimated) > 0 else None)
                    entry['test'] = {'n': len(actual), 'pearson_r': correlation,
                                     'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt(np.mean(error ** 2)))}
                    r_label = f'{correlation:.3f}' if correlation is not None else 'undefined'
                    scatter_axis.text(.04, .96, f'n={len(actual)}\nr={r_label}\nRMSE={entry["test"]["rmse"]:.2f}\nMAE={entry["test"]["mae"]:.2f}',
                                      transform=scatter_axis.transAxes, va='top', fontsize=6)
                else:
                    entry['test'] = {'n': 0}
                    scatter_axis.text(.5, .5, 'No observed test labels', transform=scatter_axis.transAxes, ha='center')
                scatter_axis.set(xlabel=f'Observed {display_name}', ylabel=f'Predicted {display_name}')
                scatter_axis.set_title(display_name)
                _axis_style(scatter_axis)
                scatter.tight_layout()
                _save(scatter, destination / f'target-{index + 1:03d}_test-scatter')
            else:
                probabilities = np.asarray([[float(row[f'{name}__prob_{category:03d}'])
                                             for category in range(width)] for row in aligned])[valid]
                entry['test'] = {'n': int(valid.sum()), 'roc_auc': _roc_figure(
                    observed[valid], probabilities, classes, display_name,
                    destination / f'target-{index + 1:03d}_test-roc')}
            summary['targets'][name] = entry
            offset += width
        figure.tight_layout(h_pad=1.5)
        _save(figure, destination / 'latent_weights')
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return destination
