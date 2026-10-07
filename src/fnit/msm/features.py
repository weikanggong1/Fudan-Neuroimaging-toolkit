"""HCP MSMAll regression and explicit, grid-matched C/A/T input preparation.

The regression follows HCPpipelines v4.7.0 ``MSMregression.m``.  PyTorch
performs the regression; Workbench performs the specified 14 mm CIFTI
smoothing when the WRN spatial weights are requested.  Subject myelin and
variance normalization are supplied inputs, never inferred from a template.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import nibabel as nib
import numpy as np
import torch


HCP_FEATURE_COMMIT = "f8cac6892f88bdf889d644711ff038198eb81533"


@dataclass(frozen=True)
class MSMAllRegressionResult:
    spatial_maps: Path
    component_weights: Path
    node_timeseries: Path
    report: Path


def _demean(values, axis):
    return values - values.mean(dim=axis, keepdim=True)


def _pinv(values):
    # MATLAB pinv uses max(size(A))*eps*largest_singular_value.
    return torch.linalg.pinv(values, rtol=max(values.shape) * np.finfo(np.float64).eps)


def _node_timeseries(data, maps, weights=None, *, observations=None):
    if weights is None:
        design = _demean(maps, 0)
        if observations is None:
            observations = _demean(data, 0)
    else:
        root = torch.sqrt(weights)[:, None]
        # The source demeans the weighted design, but weights already
        # spatially demeaned observations. These orders are different.
        design = _demean(maps * root, 0)
        if observations is None:
            observations = _demean(data, 0) * root
    return _demean((_pinv(design) @ observations).T, 0)


def _spatial_maps(data, node_timeseries, *, observations=None):
    if observations is None:
        observations = _demean(data.T, 0)
    return (_pinv(node_timeseries) @ observations).T


def _regression(data, reference_maps, *, method, cortical_area=None,
                low_maps=(), smooth=None):
    """Literal double-precision operator on [grayordinate, time/component]."""
    if method == "DR":
        nodes = _node_timeseries(data, reference_maps)
        return _spatial_maps(data, nodes), nodes, None
    if method != "WRN":
        raise ValueError("method must be DR or WRN")
    if cortical_area is None or len(low_maps) != 15 or smooth is None:
        raise ValueError("WRN requires cortical vertex areas, d7–d21 maps and 14 mm smoothing")
    area = torch.ones(len(data), dtype=data.dtype, device=data.device)
    area[:len(cortical_area)] = cortical_area
    # All d7--d21 passes use the same complete BOLD and cortical area.
    # Reuse these two literal reductions on CPU rather than allocating and
    # centering the large BOLD matrix in each of the 31 area-weighted passes.
    # CUDA and differentiable data keep the existing evaluation sequence.
    cache_observations = data.device.type == "cpu" and not any(
        value.requires_grad for value in (data, reference_maps, cortical_area, *low_maps))
    area_observations = (_demean(data, 0) * torch.sqrt(area)[:, None]
                         if cache_observations else None)
    temporal_observations = _demean(data.T, 0) if cache_observations else None
    correlations = []
    for low_dimensional_maps in low_maps:
        nodes = _node_timeseries(data, low_dimensional_maps, area, observations=area_observations)
        maps = _spatial_maps(data, nodes, observations=temporal_observations)
        nodes = _node_timeseries(data, maps, area, observations=area_observations)
        maps = _spatial_maps(data, nodes, observations=temporal_observations)
        x = _demean(maps, 1)
        y = _demean(low_dimensional_maps, 1)
        numerator = (x * y).sum(1)
        denominator = torch.sqrt(x.square().sum(1)) * torch.sqrt(y.square().sum(1))
        correlations.append(torch.atanh(numerator / denominator))
    # MATLAB assigns corrs(:,i) for i=4..18. The first three columns
    # therefore contain zeros and participate in mean(corrs,2).
    fisher = torch.stack(correlations, 1).sum(1) / (len(correlations) + 3)
    smoothed = smooth(fisher)
    spatial_weights = (fisher.mean() + fisher - smoothed).clamp_min(0).pow(3)
    weights = area * spatial_weights
    nodes = _node_timeseries(data, reference_maps, weights)
    maps = _spatial_maps(data, nodes, observations=temporal_observations)
    nodes = _node_timeseries(data, maps, area, observations=area_observations)
    maps = _spatial_maps(data, nodes, observations=temporal_observations)
    cortical_count = len(cortical_area)
    original = reference_maps[:cortical_count]
    generated = maps[:cortical_count]
    maps = ((maps - generated.mean(0)) / generated.std(0, correction=1)
            * original.std(0, correction=1) + original.mean(0))
    return maps, nodes, spatial_weights


def _cifti(path):
    image = nib.load(str(path))
    if not isinstance(image, nib.Cifti2Image) or len(image.shape) != 2:
        raise ValueError("MSMAll regression inputs must be two-dimensional CIFTI-2 files")
    axis = image.header.get_axis(1)
    if not isinstance(axis, nib.cifti2.BrainModelAxis):
        raise ValueError("CIFTI columns must contain a BrainModelAxis")
    values = np.asarray(image.dataobj, dtype=np.float64).T
    if not np.isfinite(values).all():
        raise ValueError("CIFTI input contains nonfinite values")
    return image, axis, values


def _same_axis(first, second):
    same = (np.array_equal(first.name, second.name)
            and np.array_equal(first.vertex, second.vertex)
            and np.array_equal(first.voxel, second.voxel)
            and first.nvertices == second.nvertices
            and first.volume_shape == second.volume_shape)
    if first.affine is None or second.affine is None:
        return same and first.affine is None and second.affine is None
    return same and np.array_equal(first.affine, second.affine)


def _bold_cifti(path):
    image, axis, values = _cifti(path)
    if not isinstance(image.header.get_axis(0), nib.cifti2.SeriesAxis) or values.shape[1] < 2:
        raise ValueError("clean_dtseries must have a CIFTI SeriesAxis and at least two timepoints")
    return image, axis, values


def _write_cifti(path, data, axis, names):
    header = nib.Cifti2Header.from_axes((nib.cifti2.ScalarAxis(names), axis))
    nib.save(nib.Cifti2Image(np.asarray(data.T, dtype=np.float32), header), str(path))


def _component_indices(value, count):
    if value is None:
        return np.arange(1, count + 1)
    if isinstance(value, (str, Path)):
        value = np.loadtxt(value, ndmin=1)
    indices = np.asarray(value)
    if (indices.ndim != 1 or len(indices) == 0 or not np.isfinite(indices).all()
            or np.any(indices != np.floor(indices)) or np.any(indices < 1)
            or np.any(indices > count) or len(np.unique(indices)) != len(indices)):
        raise ValueError("component_indices must contain unique one-based reference-map indices")
    return indices.astype(np.int64)


def compute_msmall_variance_normalization(clean_dtseries, ica_timecourses,
                                         noise_components, output_file, *, device="cuda:0"):
    """Compute HCP's unstructured-noise VN using an existing ICA classification.

    ICA mixing has [time, component] shape. Noise indices are one-based;
    they may be a sequence, a numeric text file, or the final bracketed list
    of a FIX classification file. No classification or data cleaning runs
    here. Signal components are regressed from a temporary demeaned copy;
    the sample standard deviation of its residual is floored at 0.001.
    """
    started = time.perf_counter()
    _, axis, data = _bold_cifti(clean_dtseries)
    mixing = np.loadtxt(ica_timecourses, ndmin=2) if isinstance(ica_timecourses, (str, Path)) else np.asarray(ica_timecourses)
    if (mixing.ndim != 2 or mixing.shape[0] != data.shape[1]
            or mixing.shape[1] == 0 or not np.isfinite(mixing).all()):
        raise ValueError("ICA timecourses must be finite and match the clean BOLD timepoints")
    if isinstance(noise_components, (str, Path)):
        content = Path(noise_components).read_text(encoding="utf-8")
        bracketed = re.findall(r"\[([^\[\]]*)\]", content)
        if bracketed:
            content = bracketed[-1]
        try:
            noise_components = np.asarray([float(value) for value in re.split(r"[\s,]+", content.strip()) if value])
        except ValueError as error:
            raise ValueError("noise_components must contain a numeric one-based component list") from error
    noise = np.asarray(noise_components)
    if noise.ndim != 1:
        raise ValueError("noise_components must be an explicit one-dimensional index list")
    if len(noise):
        noise = _component_indices(noise, mixing.shape[1])
    signal = np.setdiff1d(np.arange(mixing.shape[1]), noise.astype(np.int64) - 1)
    selected = torch.device(device)
    mixing_tensor = torch.as_tensor(mixing, dtype=torch.float64, device=selected)
    mixing_tensor = _demean(mixing_tensor, 0)
    # HCP global/matlab/normalise.m uses a 1e-5 SD floor.
    mixing_tensor = mixing_tensor / mixing_tensor.std(0, correction=1).clamp_min(0.00001)
    clean = _demean(torch.as_tensor(data, dtype=torch.float64, device=selected), 1)
    signal_tensor = mixing_tensor[:, signal]
    if len(signal):
        residual = clean - (signal_tensor @ (_pinv(signal_tensor) @ clean.T)).T
    else:
        residual = clean
    vn = residual.std(1, correction=1).clamp_min(0.001).detach().cpu().numpy()
    if not np.isfinite(vn).all():
        raise ValueError("variance normalization produced nonfinite values")
    output = Path(output_file).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_cifti(output, vn[:, None], axis, ["Unstructured-noise variance normalization"])
    output.with_suffix(".json").write_text(json.dumps({
        "hcp_commit": HCP_FEATURE_COMMIT, "method": "ComputeVN",
        "ica_classification_supplied": True, "classification_performed": False,
        "noise_cleanup_performed": False, "bias_field_reverted": False,
        "signal_component_count": int(len(signal)), "noise_component_count": int(len(noise)),
        "standard_deviation_correction": 1, "minimum_vn": 0.001,
        "seconds": time.perf_counter() - started, "device": str(selected),
    }, indent=2) + "\n", encoding="utf-8")
    return output


def run_msmall_regression(clean_dtseries, reference_maps, output_dir, *,
                          variance_normalization, vertex_area=None,
                          component_indices=None, low_dimensional_maps=None,
                          left_midthickness=None, right_midthickness=None,
                          method="WRN", device="cuda:0", wb_command="wb_command"):
    """Generate individual RSN/topographic maps with HCP DR or WRN.

    ``clean_dtseries`` and all reference maps must have the same grayordinate
    order. ``variance_normalization`` is the supplied HCP VN dscalar (one
    positive value per grayordinate). Passing ``None`` explicitly is allowed
    for ordinary DR; it is not HCP's default WRN preprocessing.

    WRN requires a cortical-only, mean-one vertex-area dscalar, d7–d21 maps
    in that order, and the matching left/right midthickness surfaces. The
    selected one-based component indices are those in HCP's Weights.txt.
    Output maps retain all reference columns and their names; the weight
    file selects the requested components with zero/one columns.
    """
    started = time.perf_counter()
    output = Path(output_dir).expanduser().resolve()
    if method not in ("DR", "WRN"):
        raise ValueError("method must be DR or WRN")
    _, axis, data = _bold_cifti(clean_dtseries)
    reference, reference_axis, maps = _cifti(reference_maps)
    if not _same_axis(axis, reference_axis):
        raise ValueError("clean BOLD and reference maps have different grayordinate grids")
    if not isinstance(reference.header.get_axis(0), nib.cifti2.ScalarAxis):
        raise ValueError("reference_maps must contain named CIFTI scalar maps")
    names = list(reference.header.get_axis(0).name)
    indices = _component_indices(component_indices, maps.shape[1])
    if variance_normalization is not None:
        _, vn_axis, vn = _cifti(variance_normalization)
        if not _same_axis(axis, vn_axis) or vn.shape != (len(data), 1) or np.any(vn <= 0):
            raise ValueError("variance_normalization must be a positive scalar on the BOLD grid")
        # SingleSubjectConcat.sh: demean each run, then divide by its VN.
        data = ((data - data.mean(1, keepdims=True)) / np.maximum(vn, 0.001)).astype(np.float32).astype(np.float64)
    elif method == "WRN":
        raise ValueError("WRN requires an explicitly supplied variance_normalization map")
    if method == "WRN":
        constant_count = int(np.count_nonzero(np.ptp(data, axis=1) == 0))
        if constant_count:
            raise ValueError(
                f"WRN input has {constant_count} constant BOLD time series; correlation is undefined. "
                "Explicitly subset BOLD, VN, reference maps and cortical area to the same valid BrainModelAxis.")
    selected = torch.device(device)
    data_tensor = torch.as_tensor(data, dtype=torch.float64, device=selected)
    maps_tensor = torch.as_tensor(maps, dtype=torch.float64, device=selected)
    area_tensor = None
    low_tensors = []
    smoother = None
    if method == "WRN":
        if vertex_area is None or left_midthickness is None or right_midthickness is None:
            raise ValueError("WRN requires vertex_area and both matching midthickness surfaces")
        if low_dimensional_maps is None or len(low_dimensional_maps) != 15:
            raise ValueError("WRN requires all 15 low-dimensional reference maps, d7 through d21")
        _, area_axis, area = _cifti(vertex_area)
        cortical = np.isin(axis.name, ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"))
        count = int(cortical.sum())
        if (not np.all(cortical[:count]) or np.any(cortical[count:])
                or not _same_axis(axis[:count], area_axis) or area.shape != (count, 1)
                or np.any(area <= 0) or not np.isclose(area.mean(), 1, rtol=1e-5)):
            raise ValueError("vertex_area must contain positive mean-one cortical areas in BOLD order")
        area_tensor = torch.as_tensor(area[:, 0], dtype=torch.float64, device=selected)
        for dimensionality, path in zip(range(7, 22), low_dimensional_maps):
            _, low_axis, low = _cifti(path)
            if not _same_axis(axis, low_axis) or low.shape[1] != dimensionality:
                raise ValueError(f"d{dimensionality} reference map has a different grid or dimensionality")
            low_tensors.append(torch.as_tensor(low, dtype=torch.float64, device=selected))
        executable = shutil.which(str(wb_command))
        if executable is None:
            raise FileNotFoundError(f"Connectome Workbench not found: {wb_command}")
        for surface in (left_midthickness, right_midthickness):
            if not Path(surface).is_file():
                raise FileNotFoundError(surface)
        output.mkdir(parents=True, exist_ok=True)

        def smoother(values):
            # Official MATLAB writes float32 CIFTI before Workbench smoothing.
            with tempfile.TemporaryDirectory(prefix="msmall-wrn-", dir=output) as temporary:
                temporary = Path(temporary)
                incoming = temporary / "weights.dscalar.nii"
                outgoing = temporary / "smoothed.dscalar.nii"
                _write_cifti(incoming, values.detach().cpu().numpy()[:, None], axis, ["WRN Fisher weight"])
                subprocess.run([executable, "-cifti-smoothing", str(incoming), "14", "14",
                                "COLUMN", str(outgoing), "-left-surface", str(left_midthickness),
                                "-right-surface", str(right_midthickness)],
                               check=True, capture_output=True, text=True)
                _, smoothed_axis, smoothed = _cifti(outgoing)
                if not _same_axis(axis, smoothed_axis):
                    raise ValueError("WRN smoothing changed the grayordinate grid")
                return torch.as_tensor(smoothed[:, 0], dtype=torch.float64, device=selected)

    individual, nodes, spatial_weights = _regression(
        data_tensor, maps_tensor, method=method, cortical_area=area_tensor,
        low_maps=low_tensors, smooth=smoother)
    individual = individual.detach().cpu().numpy()
    nodes = nodes.detach().cpu().numpy()
    if not np.isfinite(individual).all() or not np.isfinite(nodes).all():
        raise ValueError("MSMAll regression produced nonfinite maps or time series")
    output.mkdir(parents=True, exist_ok=True)
    map_path = output / "individual_maps.dscalar.nii"
    weight_path = output / "component_weights.dscalar.nii"
    node_path = output / "node_timeseries.tsv"
    flags = np.zeros(maps.shape[1], np.float64)
    flags[indices - 1] = 1
    _write_cifti(map_path, individual, axis, names)
    _write_cifti(weight_path, np.broadcast_to(flags, maps.shape), axis, names)
    np.savetxt(node_path, nodes, fmt="%.17g", delimiter="\t")
    report_path = output / "regression.json"
    report = {"method": method, "hcp_commit": HCP_FEATURE_COMMIT,
              "variance_normalization_supplied": variance_normalization is not None,
              "variance_normalization_inferred": False, "precision": "float64 regression / float32 CIFTI",
              "low_dimensional_reference_count": len(low_tensors),
              "wrn_smoothing_sigma_mm": 14 if method == "WRN" else None,
              "matlab_leading_zero_correlation_columns": 3 if method == "WRN" else 0,
              "selected_component_count": int(len(indices)),
              "finite_outputs": True, "seconds": time.perf_counter() - started,
              "device": str(selected)}
    if spatial_weights is not None:
        weights_cpu = spatial_weights.detach().cpu().numpy()
        report["spatial_weight_min"] = float(weights_cpu.min())
        report["spatial_weight_max"] = float(weights_cpu.max())
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return MSMAllRegressionResult(map_path, weight_path, node_path, report_path)


def _metric(path):
    image = nib.load(str(path))
    if not isinstance(image, nib.GiftiImage) or not image.darrays:
        raise ValueError("MSMAll per-hemisphere feature inputs must be GIFTI metrics")
    columns = [np.asarray(array.data, dtype=np.float32) for array in image.darrays]
    if any(array.ndim != 1 for array in columns):
        raise ValueError("GIFTI feature arrays must contain one column each")
    values = np.stack(columns, 1)
    if not np.isfinite(values).all():
        raise ValueError("MSMAll feature input contains nonfinite values")
    return values, [array.meta.get("Name", "") for array in image.darrays]


def _save_metric(path, values, names):
    arrays = [nib.gifti.GiftiDataArray(np.asarray(values[:, i], dtype=np.float32),
              intent="NIFTI_INTENT_NONE", meta=nib.gifti.GiftiMetaData({"Name": name}))
              for i, name in enumerate(names)]
    nib.save(nib.GiftiImage(darrays=arrays), str(path))


def _weights(path, shape):
    values, _ = _metric(path)
    if values.shape != shape or np.any(values < 0):
        raise ValueError("feature weights must be nonnegative and match every feature column")
    return values


def _source_stdev(values):
    # Workbench metric-stats STDEV is population SD, returns float32, then
    # prints seven significant digits consumed by the HCP shell/bc code.
    standard_deviation = np.std(values.astype(np.float64), axis=0, ddof=0).astype(np.float32)
    return np.asarray([float(format(float(value), ".7g")) for value in standard_deviation])


def prepare_msmall_inputs(source_sphere, reference_sphere, output_dir, *,
                         source_rsn, reference_rsn, source_rsn_weights,
                         reference_rsn_weights, subject_myelin=None, reference_myelin=None,
                         subject_myelin_bias=None, source_roi, reference_roi, modalities="CA",
                         initial_sphere=None, source_topography=None,
                         reference_topography=None, source_topography_weights=None,
                         reference_topography_weights=None):
    """Prepare one hemisphere's HCP C/A or C/A/T feature and weight files.

    Each source metric must already use ``source_sphere`` vertex order; each
    reference metric must use ``reference_sphere`` order. ``modalities`` is
    explicitly C, CA (default), or CAT. CA/CAT require subject myelin and its
    per-iteration bias field. Resampling and the HCP sqrt(200) mm myelin-bias
    smoothing belong to the outer pipeline. C never silently replaces CA.

    This performs HCP's modality scaling, appends the inverse-ROI channel,
    applies weights and removes columns whose reference weights are zero.
    It returns ``MSMAllInputs`` without performing registration.
    """
    from .msmall import MSMAllInputs

    if modalities not in ("C", "CA", "CAT"):
        raise ValueError("modalities must be C, CA or CAT")
    architecture = (subject_myelin, reference_myelin, subject_myelin_bias)
    if "A" in modalities and any(value is None for value in architecture):
        raise ValueError("CA/CAT requires explicit subject myelin, reference myelin and subject myelin bias")
    if modalities == "C" and any(value is not None for value in architecture):
        raise ValueError("C modalities do not use myelin inputs; select CA or CAT explicitly")
    topo = (source_topography, reference_topography, source_topography_weights, reference_topography_weights)
    if any(value is not None for value in topo) and any(value is None for value in topo):
        raise ValueError("topographic source/reference maps and both weights must be provided together")
    if "T" in modalities and any(value is None for value in topo):
        raise ValueError("CAT requires explicit source/reference topographic maps and both weights")
    if "T" not in modalities and any(value is not None for value in topo):
        raise ValueError("topographic inputs require modalities='CAT'")

    def mesh_geometry(path):
        image = nib.load(str(path))
        if not isinstance(image, nib.GiftiImage):
            raise ValueError("MSMAll spheres must be GIFTI surfaces")
        points = [array.data for array in image.darrays if array.intent == 1008]
        if len(points) != 1 or np.asarray(points[0]).ndim != 2 or np.asarray(points[0]).shape[1] != 3:
            raise ValueError("MSMAll sphere must contain one three-dimensional pointset")
        faces = [array.data for array in image.darrays if array.intent == 1009]
        if (len(faces) != 1 or np.asarray(faces[0]).ndim != 2
                or np.asarray(faces[0]).shape[1] != 3):
            raise ValueError("MSMAll sphere must contain one triangular topology")
        return np.asarray(points[0]), np.asarray(faces[0])

    source_points, source_faces = mesh_geometry(source_sphere)
    reference_points, _ = mesh_geometry(reference_sphere)
    source_count, reference_count = len(source_points), len(reference_points)
    if initial_sphere is not None:
        initial_points, initial_faces = mesh_geometry(initial_sphere)
        if len(initial_points) != source_count or not np.array_equal(initial_faces, source_faces):
            raise ValueError("initial_sphere must use the source feature topology")
    source, source_names = _metric(source_rsn)
    reference, reference_names = _metric(reference_rsn)
    if (source.shape[0] != source_count or reference.shape[0] != reference_count
            or source.shape[1] != reference.shape[1]):
        raise ValueError("RSN feature columns or sphere vertex counts differ")
    if all(source_names) and all(reference_names) and source_names != reference_names:
        raise ValueError("source and reference RSN component names/order differ")
    source_weights = _weights(source_rsn_weights, source.shape)
    reference_weights = _weights(reference_rsn_weights, reference.shape)
    sr, _ = _metric(source_roi)
    rr, _ = _metric(reference_roi)
    if (sr.shape != (source_count, 1) or rr.shape != (reference_count, 1)
            or np.any((sr != 0) & (sr != 1)) or np.any((rr != 0) & (rr != 1))):
        raise ValueError("source and reference ROIs must be matching binary scalar metrics")
    selected_count = int(np.count_nonzero(np.any(reference_weights > 0, axis=0)))
    cscale = float(_source_stdev(reference * reference_weights).sum() / max(selected_count, 1))
    if cscale <= 0:
        raise ValueError("reference RSNs have no positive weighted standard deviation")
    source_parts = [(source.astype(np.float64) / cscale).astype(np.float32)]
    reference_parts = [(reference.astype(np.float64) / cscale).astype(np.float32)]
    source_weight_parts = [source_weights]
    reference_weight_parts = [reference_weights]
    names = [name or f"RSN {i + 1}" for i, name in enumerate(reference_names)]
    ascale = None
    if "A" in modalities:
        myelin, _ = _metric(subject_myelin)
        atlas_myelin, _ = _metric(reference_myelin)
        bias, _ = _metric(subject_myelin_bias)
        if (myelin.shape != sr.shape or bias.shape != sr.shape or atlas_myelin.shape != rr.shape):
            raise ValueError("subject myelin, per-iteration bias and atlas myelin must be scalar on their own sphere")
        ascale = float(_source_stdev(atlas_myelin * rr)[0])
        if ascale <= 0:
            raise ValueError("reference myelin has no positive ROI-weighted standard deviation")
        source_parts.append(((myelin.astype(np.float64) - bias) / ascale).astype(np.float32))
        reference_parts.append((atlas_myelin.astype(np.float64) / ascale).astype(np.float32))
        source_weight_parts.append(sr)
        reference_weight_parts.append(rr)
        names.append("Myelin")
    tscale = None
    if "T" in modalities:
        ts, ts_names = _metric(source_topography)
        tr, tr_names = _metric(reference_topography)
        if ts.shape[0] != source_count or tr.shape[0] != reference_count or ts.shape[1] != tr.shape[1]:
            raise ValueError("topographic feature columns or sphere vertex counts differ")
        if all(ts_names) and all(tr_names) and ts_names != tr_names:
            raise ValueError("source and reference topographic map names/order differ")
        tws, twr = _weights(source_topography_weights, ts.shape), _weights(reference_topography_weights, tr.shape)
        # The HCP shell sums the per-map STDEVs and divides by 1 here.
        tscale = float(_source_stdev(tr * twr).sum())
        if tscale <= 0:
            raise ValueError("reference topography has no positive weighted standard deviation")
        source_parts.append((ts.astype(np.float64) / tscale).astype(np.float32))
        reference_parts.append((tr.astype(np.float64) / tscale).astype(np.float32))
        source_weight_parts.append(tws)
        reference_weight_parts.append(twr)
        names.extend(name or f"Topography {i + 1}" for i, name in enumerate(tr_names))
    source_parts.append(1 - sr)
    reference_parts.append(1 - rr)
    source_weight_parts.append(1 - sr)
    reference_weight_parts.append(1 - rr)
    names.append("Medial wall")
    sf, rf = np.concatenate(source_parts, 1), np.concatenate(reference_parts, 1)
    sw, rw = np.concatenate(source_weight_parts, 1), np.concatenate(reference_weight_parts, 1)
    # Workbench metric-math evaluates in double and writes each result as float.
    sf = (sf.astype(np.float64) * sw * cscale).astype(np.float32)
    rf = (rf.astype(np.float64) * rw * cscale).astype(np.float32)
    keep = np.any(rw != 0, axis=0)
    names = [name for name, selected in zip(names, keep) if selected]
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in ("source_features.func.gii", "reference_features.func.gii",
                                       "source_weights.func.gii", "reference_weights.func.gii")]
    for path, values in zip(paths, (sf[:, keep], rf[:, keep], sw[:, keep], rw[:, keep])):
        _save_metric(path, values, names)
    (output / "features.json").write_text(json.dumps({
        "hcp_commit": HCP_FEATURE_COMMIT, "modalities": modalities,
        "subject_myelin_supplied": subject_myelin is not None, "subject_myelin_inferred": False,
        "myelin_bias_supplied": subject_myelin_bias is not None, "rsn_scale": cscale, "myelin_scale": ascale,
        "topography_scale": tscale, "feature_columns": len(names),
        "dropped_zero_reference_weight_columns": int((~keep).sum()),
    }, indent=2) + "\n", encoding="utf-8")
    return MSMAllInputs(source_sphere=Path(source_sphere), source_features=paths[0],
                        reference_sphere=Path(reference_sphere), reference_features=paths[1],
                        initial_sphere=None if initial_sphere is None else Path(initial_sphere),
                        source_weights=paths[2], reference_weights=paths[3])
