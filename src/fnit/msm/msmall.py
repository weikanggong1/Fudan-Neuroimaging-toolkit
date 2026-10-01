"""Independent HCP MSMAll vector-feature registration using FNIT HOCR/FastPD."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import json
import time

import nibabel as nib
import numpy as np
import torch

from ._affine import _surface
from ._sphere_map import RadialSphereMap
from .config_msmall import MSMAllConfig
from .msmsulc import (
    _adaptive_resample, _face_layout, _ico, _label_samples, _native_output_qc,
    _normalize_sphere, _regularized_triangle_cost, _rescaled_labels,
    _rotated_label, _rotation_matrices, _sphere_warp, _triplet_data_weights,
    _unfold, _unit3, _variance_normalize, _vertex_area,
)

_adaptive_resample = partial(_adaptive_resample, source_precision=True)
_sphere_warp = partial(_sphere_warp, source_precision=True)


@dataclass(frozen=True)
class MSMAllInputs:
    """Features on ``source_sphere`` and a separate previous registration.

    ``initial_sphere`` corresponds to official ``--trans``; it must preserve
    the source vertex order and topology. Each feature GIFTI has one column
    per map. Optional cost-weight GIFTIs have one or the feature count columns.
    The two input feature matrices must have the same number of columns.
    """

    source_sphere: str | Path
    source_features: str | Path
    reference_sphere: str | Path
    reference_features: str | Path
    initial_sphere: str | Path | None = None
    source_weights: str | Path | None = None
    reference_weights: str | Path | None = None


def _read_features(path, vertex_count):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage) or not image.darrays:
        raise ValueError("MSMAll features must be a nonempty GIFTI metric")
    columns = []
    for array in image.darrays:
        values = np.asarray(array.data, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != vertex_count:
            raise ValueError("MSMAll metric must have one row per sphere vertex")
        if array.intent in (1008, 1009):
            raise ValueError("MSMAll features cannot contain surface geometry")
        columns.append(values)
    values = np.concatenate(columns, axis=1)
    if not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("MSMAll metric must contain finite feature columns")
    return values


def _variance_normalize_columns(values):
    return np.column_stack([_variance_normalize(values[:, column])
                            for column in range(values.shape[1])])


def _true_rescale(vertices):
    """Point's radius-only normalization, without another recenter operation."""
    squared = (vertices[:, 0] * vertices[:, 0] + vertices[:, 1] * vertices[:, 1]) + vertices[:, 2] * vertices[:, 2]
    norm = np.sqrt(squared)
    if np.any(norm <= 1e-8):
        raise ValueError("sphere contains a zero-radius vertex")
    return vertices / norm[:, None] * 100


def _same_sphere_coordinates(first, second):
    """newMSM Mesh equality delegates to Point's strict 1e-8 tolerance."""
    return first.shape == second.shape and np.all(np.abs(first - second) < 1e-8)


def _nearest_weights(vertices, faces, values, grid, device, execution):
    # Official nearest interpolation first selects the containing triangle;
    # it then compares only its corners, in original mesh order.
    from . import _fastpd_native
    mapper = RadialSphereMap(vertices, faces, device, execution=execution, source_precision=True)
    query = np.asarray(grid, dtype=np.float64)
    _, _, patches = mapper.weights(torch.as_tensor(query, device=device))
    ids = np.frombuffer(_fastpd_native.source_triangle_nearest(
        mapper.vertex_bytes, mapper.face_bytes, query.tobytes(),
        patches.detach().cpu().numpy().astype(np.int64, copy=False).tobytes(),
        len(vertices), len(faces), len(grid)), dtype=np.int64)
    return values[ids]


def _combine_weights(source, reference):
    """Official row overlap average; remaining higher-dimensional rows persist."""
    result = (source if source.shape[1] >= reference.shape[1] else reference).copy()
    overlap = min(source.shape[1], reference.shape[1])
    result[:, :overlap] = (source[:, :overlap] + reference[:, :overlap]) / 2.0
    return result


def _feature_weights(values, dimensions):
    # newMSM does not broadcast a one-row matrix to every feature. Rows absent
    # from HIGHREScfweight have weight one in HOMultivariate::get_source_data.
    result = np.ones((len(values), dimensions), dtype=np.float64)
    result[:, :values.shape[1]] = values
    return result


def _weighted_vector_cost(source, target, weights):
    """Weighted Pearson across features independently at each DATA vertex."""
    total = weights.sum(-1)
    positive = total > 0
    safe = torch.where(positive, total, torch.ones_like(total))
    mean_source = (weights * source).sum(-1)
    mean_target = (weights * target).sum(-1)
    mean_source = torch.where(positive, mean_source / safe, mean_source)
    mean_target = torch.where(positive, mean_target / safe, mean_target)
    first = source - mean_source[..., None]
    second = target - mean_target[..., None]
    covariance = (weights * first * second).sum(-1)
    var_source = (weights * first * first).sum(-1)
    var_target = (weights * second * second).sum(-1)
    covariance = torch.where(positive, covariance / safe, covariance)
    var_source = torch.where(positive, var_source / safe, var_source)
    var_target = torch.where(positive, var_target / safe, var_target)
    denominator = torch.sqrt(var_source) * torch.sqrt(var_target)
    correlation = torch.where((var_source != 0) & (var_target != 0),
                              covariance / torch.where(denominator != 0, denominator, 1), 0)
    return 1 - (1 + correlation) * 0.5


def _multivariate_face_costs(current, candidate, original, faces, layout,
                             reference_map, reference_features, feature_weights,
                             absolute_weights, lam, config, *, energy_only=False,
                             fold_reference=None):
    index, valid, weights, source, packed = layout
    bits = torch.as_tensor([[i >> 2 & 1, i >> 1 & 1, i & 1]
                            for i in range(1 if energy_only else 8)],
                           dtype=torch.bool, device=current.device)
    proposed = torch.where(bits[None, :, :, None], candidate[faces][:, None],
                           current[faces][:, None])
    weighted = weights[index][:, None, :, :, None] * proposed[:, :, None]
    points = _unit3((weighted[:, :, :, 0] + weighted[:, :, :, 1]) + weighted[:, :, :, 2]) * 100
    states = len(bits); width = index.shape[1]
    gathered = ((packed // width)[:, None] * (states * width) +
                torch.arange(states, device=current.device)[None, :] * width +
                (packed % width)[:, None]).reshape(-1)
    ids, barycentric_weights, _ = reference_map.weights(points.reshape(-1, 3)[gathered], project=False)
    corner_values = reference_features[ids] * barycentric_weights[:, :, None]
    sampled = (corner_values[:, 0] + corner_values[:, 1]) + corner_values[:, 2]
    target = torch.zeros((*points.shape[:-1], source.shape[1]),
                         dtype=source.dtype, device=current.device)
    target.reshape(-1, source.shape[1])[gathered] = sampled
    local_cost = _weighted_vector_cost(source[index][:, None], target,
                                       feature_weights[index][:, None])
    count = valid.sum(-1).clamp_min(1)
    similarity = (local_cost * valid[:, None]).sum(-1) / count[:, None]
    cp_weight = absolute_weights[faces]
    cp_weight = ((cp_weight[:, 0] + cp_weight[:, 1]) + cp_weight[:, 2]) / 3.0
    similarity = cp_weight[:, None] * similarity
    return _regularized_triangle_cost(similarity, proposed, original, faces, lam,
                                       config, current, fold_reference).detach().cpu().numpy()


def _register_msmall_one(entry, output_dir, *, hemi="L", device="cuda:0", config=None,
                         execution="optimized"):
    """One hemisphere entry point for paired checkpoint diagnostics."""
    from . import _fastpd_native
    if not isinstance(entry, MSMAllInputs):
        raise TypeError("MSMAll inputs must be MSMAllInputs")
    if config is None:
        config = MSMAllConfig()
    elif isinstance(config, (str, Path)):
        config = MSMAllConfig.from_file(config)
    if not isinstance(config, MSMAllConfig):
        raise TypeError("config must be MSMAllConfig or a config path")
    if execution not in ("optimized", "reference"):
        raise ValueError("execution must be optimized or reference")
    if hemi not in ("L", "R"):
        raise ValueError("hemi must be L or R")
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(selected)
    output = Path(output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    native, native_faces = _surface(entry.source_sphere)
    reference, reference_faces = _surface(entry.reference_sphere)
    native_area = _vertex_area(native, native_faces)
    reference_area = _vertex_area(reference, reference_faces)
    native = _normalize_sphere(native); reference = _normalize_sphere(reference)
    initial = None
    if entry.initial_sphere is not None:
        initial, initial_faces = _surface(entry.initial_sphere)
        if initial.shape != native.shape or not np.array_equal(initial_faces, native_faces):
            raise ValueError("initial_sphere must preserve source vertex order and topology")
        # set_transformed() loads the sphere unchanged and applies true_rescale
        # again to the reference mesh; preserve that upstream behavior.
        reference = _true_rescale(reference)
    source_features = _read_features(entry.source_features, len(native))
    reference_features = _read_features(entry.reference_features, len(reference))
    dimensions = source_features.shape[1]
    if dimensions < 2 or reference_features.shape[1] != dimensions:
        raise ValueError("MSMAll requires matching source/reference feature counts of at least two")
    source_weights = None if entry.source_weights is None else _read_features(entry.source_weights, len(native))
    reference_weights = None if entry.reference_weights is None else _read_features(entry.reference_weights, len(reference))
    for values in (source_weights, reference_weights):
        if values is not None and values.shape[1] not in (1, dimensions):
            raise ValueError("cost weights must have one or the feature count columns")
    weighted_cost = source_weights is not None and reference_weights is not None
    stages = []; previous_grid = previous_faces = previous_positions = None
    for stage_index, level in enumerate(config.control_grid):
        stage_started = time.perf_counter()
        regular_np, faces_np = _ico(level)
        data_np, data_faces, data_area = _ico(config.data_grid[stage_index], cached_area=True)
        # Feature resampling retains the construction-time triangle areas.
        # In contrast, Mesh copies used by cost-weight resampling reconstruct
        # their triangles after radius normalization (and, below, deformation).
        data_mesh_area = _vertex_area(data_np, data_faces)
        label_grid, label_faces = _ico(config.sampling_grid[stage_index])
        regular = torch.as_tensor(regular_np, dtype=torch.float64, device=selected)
        if stage_index == 0:
            if initial is None or _same_sphere_coordinates(initial, native):
                source_positions = torch.as_tensor(data_np, dtype=torch.float64, device=selected)
                cp_positions = regular.clone()
            else:
                initial_positions = torch.as_tensor(initial, dtype=torch.float64, device=selected)
                source_positions = _sphere_warp(torch.as_tensor(data_np, device=selected), native,
                                               native_faces, initial_positions, selected, execution=execution)
                cp_positions = _sphere_warp(regular, native, native_faces,
                                           initial_positions, selected, execution=execution)
        else:
            native_positions = _sphere_warp(torch.as_tensor(native, device=selected), previous_grid,
                                            previous_faces, previous_positions, selected, execution=execution)
            source_positions = _sphere_warp(torch.as_tensor(data_np, device=selected), native,
                                            native_faces, native_positions, selected, execution=execution)
            cp_positions = _sphere_warp(regular, native, native_faces, native_positions,
                                       selected, execution=execution)
        source_positions, source_unfold = _unfold(source_positions, data_faces)
        cp_positions, control_unfold = _unfold(cp_positions, faces_np)
        source_values = _variance_normalize_columns(_adaptive_resample(
            native, native_faces, source_features, data_np, data_faces, device=selected,
            execution=execution, old_area=native_area, new_area=data_area))
        target_values = _variance_normalize_columns(_adaptive_resample(
            reference, reference_faces, reference_features, data_np, data_faces, device=selected,
            execution=execution, old_area=reference_area, new_area=data_area))
        source_values = torch.as_tensor(source_values, device=selected)
        target_values = torch.as_tensor(target_values, device=selected)
        if weighted_cost:
            source_weight_grid = _nearest_weights(_true_rescale(native), native_faces, source_weights,
                                                 data_np, selected, execution)
            target_weight_grid = _nearest_weights(_true_rescale(reference), reference_faces, reference_weights,
                                                 data_np, selected, execution)
        target_map = RadialSphereMap(data_np, data_faces, selected, execution=execution, source_precision=True)
        edges = np.unique(np.sort(np.concatenate((faces_np[:, [0, 1]], faces_np[:, [1, 2]],
                                                 faces_np[:, [2, 0]])), axis=1), axis=0)
        chord = np.linalg.norm(regular_np[edges[:, 0]] - regular_np[edges[:, 1]], axis=1)
        spacing = float((200 * np.arcsin(chord / 200)).max())
        centre, samples = _label_samples(label_grid, label_faces, spacing * 0.5)
        face_tensor = torch.as_tensor(np.sort(faces_np, axis=1), device=selected)
        face_bytes = face_tensor.detach().cpu().numpy().astype(np.int32).tobytes()
        strain_original = torch.as_tensor(data_np[:len(regular_np)], device=selected)
        lam = config.regularization[stage_index]
        iterations = []; scale = 1.0; previous_energy = 0.0; converged = False
        for iteration in range(config.iterations[stage_index]):
            iteration_started = time.perf_counter()
            prior = cp_positions.clone(); prior_np = prior.detach().cpu().numpy()
            source_np = source_positions.detach().cpu().numpy()
            if weighted_cost:
                resampled_target_weights = _adaptive_resample(
                    data_np, data_faces, target_weight_grid, source_np, data_faces, device=selected,
                    execution=execution, old_area=data_mesh_area, new_area=data_mesh_area)
                combined = _combine_weights(source_weight_grid, resampled_target_weights)
            else:
                # The official implementation only combines provided weights
                # when both --inweight and --refweight are present.
                combined = np.ones((len(data_np), 1), dtype=np.float64)
            absolute = _adaptive_resample(source_np, data_faces, combined.max(1), prior_np,
                                          faces_np, device=selected, execution=execution,
                                          old_area=_vertex_area(source_np, data_faces),
                                          new_area=_vertex_area(prior_np, faces_np))
            absolute = torch.as_tensor(absolute, device=selected)
            feature_weights = torch.as_tensor(_feature_weights(combined, dimensions), device=selected)
            rotations = _rotation_matrices(prior_np, centre, selected)
            current_map = RadialSphereMap(prior_np, faces_np, selected, execution=execution, source_precision=True)
            _, _, patch = current_map.weights(source_positions)
            weights = _triplet_data_weights(prior, face_tensor, patch, source_positions)
            layout = _face_layout(face_tensor.detach().cpu().numpy(), patch, weights, source_values, selected)
            labels = np.zeros(len(regular_np), np.int16); changed = 0
            label_positions, scale = _rescaled_labels(centre, np.vstack((centre, samples)), scale)
            cp_positions = _rotated_label(rotations, label_positions[0])
            for _ in range(2):
                for label, sample in enumerate(label_positions):
                    if np.all(labels == label):
                        continue
                    candidate = _rotated_label(rotations, sample)
                    costs = _multivariate_face_costs(
                        cp_positions, candidate, strain_original, face_tensor, layout, target_map,
                        target_values, feature_weights, absolute, lam, config, fold_reference=prior)
                    choice = np.frombuffer(_fastpd_native.optimize(
                        face_bytes, costs.astype(np.float64).tobytes(), len(regular_np)), dtype=np.uint8)
                    update = (choice == 1) & (labels != label)
                    if update.any():
                        mask = torch.as_tensor(update, device=selected)
                        cp_positions[mask] = candidate[mask]; labels[update] = label
                        changed += int(update.sum())
            energy_costs = _multivariate_face_costs(
                cp_positions, cp_positions, strain_original, face_tensor, layout, target_map,
                target_values, feature_weights, absolute, lam, config,
                energy_only=True, fold_reference=prior)
            energy = sum(float(value) for value in energy_costs[:, 0])
            stopping = iteration > 2 and (iteration - 1) % 2 == 0 and previous_energy - energy < 0.001
            if stopping:
                cp_positions = prior; converged = True
                iterations.append({"changed": changed, "energy": energy, "applied": False,
                                   "seconds": time.perf_counter() - iteration_started})
                break
            source_positions = _sphere_warp(source_positions, prior_np, faces_np, cp_positions,
                                            selected, execution=execution)
            cp_positions, moved = _unfold(cp_positions, faces_np); control_unfold += moved
            source_positions, moved = _unfold(source_positions, data_faces); source_unfold += moved
            previous_energy = energy
            iterations.append({"changed": changed, "energy": energy, "applied": True,
                               "seconds": time.perf_counter() - iteration_started})
        previous_grid = data_np; previous_faces = data_faces; previous_positions = source_positions
        stages.append({"control_points": len(regular_np), "data_points": len(data_np),
                       "labels": len(samples) + 1, "similarity": config.simval[stage_index],
                       "maximum_iterations": config.iterations[stage_index], "converged": converged,
                       "source_unfold_updates": source_unfold, "control_unfold_updates": control_unfold,
                       "iterations": iterations, "seconds": time.perf_counter() - stage_started})
    vertices = _sphere_warp(torch.as_tensor(native, device=selected), previous_grid, previous_faces,
                            previous_positions, selected, execution=execution).detach().cpu().numpy()
    qc = _native_output_qc(vertices, native_faces, native)
    path = output / f"{hemi}.sphere.MSMAll.native.surf.gii"
    nib.save(nib.GiftiImage(darrays=[
        nib.gifti.GiftiDataArray(vertices.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
        nib.gifti.GiftiDataArray(native_faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE")]), path)
    report = {"seconds": time.perf_counter() - started, "feature_count": dimensions,
              "weighted_cost": weighted_cost, "config": config.to_dict(), "execution": execution,
              **qc, "stages": stages,
              "peak_allocated_gb": torch.cuda.max_memory_allocated(selected) / 1e9
              if selected.type == "cuda" else None}
    return path, report


def run_msmall(inputs, output_dir, *, device="cuda:0", config=None, execution="optimized"):
    """Register bilateral feature matrices; return the two native-order spheres."""
    if set(inputs) != {"L", "R"}:
        raise ValueError("inputs must contain L and R MSMAll inputs")
    paths = {}; report = {}
    for hemi in "LR":
        paths[hemi], report[hemi] = _register_msmall_one(
            inputs[hemi], output_dir, hemi=hemi, device=device, config=config, execution=execution)
    (Path(output_dir).expanduser().resolve() / "registration_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return paths
