"""Original B shared-sampler score with isolated old package and new paths.

No registration, optimizer, official command or changed gate is present.
"""
from pathlib import Path
from run_real_worker import digest

def header_affine(image, legacy):
    # Do not use the registration input's orthogonality guard on a final
    # affine mapmovhdr output, whose directions may contain scale/shear.
    import numpy as np
    from importlib import import_module
    _compose_native = import_module(legacy.__name__ + ".registration")._compose_native
    return _compose_native(image.shape, np.asarray(image.header["delta"], np.float32),
                           np.asarray(image.header["Mdc"], np.float32).T,
                           np.asarray(image.header["Pxyz_c"], np.float32))

def score(plan, directory):
    import nibabel as nib
    import numpy as np
    import torch
    from fnit._transforms import load_lta
    from load_experiment import load_package
    legacy = load_package(plan, candidate=False)
    from importlib import import_module
    sampling = import_module(legacy.__name__ + '._sampling')
    (native_inverse, native_matmul, resample) = (sampling.native_inverse, sampling.native_matmul, sampling.resample)
    out = Path(plan['run_directory'])
    (old, new) = (Path(plan['official_directory']), out / 'stages')
    (original, target) = (nib.load(plan['moving']), nib.load(plan['fixed']))
    values = np.asanyarray(original.dataobj)
    fixed_mask = torch.from_numpy(np.array(target.dataobj, dtype=np.float32, copy=True)) > 0
    if not bool(fixed_mask.any()) or float(values.max()) <= 0:
        raise ValueError('original moving and fixed masks must be nonempty')
    mask_threshold = float(values.max()) * 0.5

    def overlap(warped):
        moving_mask = warped > mask_threshold
        (nsource, ntarget) = (int(moving_mask.sum()), int(fixed_mask.sum()))
        intersection = int((moving_mask & fixed_mask).sum())
        return {'moving_threshold': mask_threshold, 'fixed_threshold': 0.0, 'moving_voxels': nsource, 'fixed_voxels': ntarget, 'intersection_voxels': intersection, 'Dice': 2 * intersection / (nsource + ntarget), 'scoring_only_not_optimization_feedback': True}
    initial_pull = native_matmul(native_inverse(header_affine(original, legacy)), header_affine(target, legacy))
    initial_warp = resample(torch.from_numpy(values.astype(np.float32)), target.shape, initial_pull, chunk_size=plan['parameters']['spatial_chunk_size'])
    initial_overlap = overlap(initial_warp)
    shape = np.asarray(original.shape)
    origin = header_affine(original, legacy).astype(float)
    corners = np.array([[x, y, z] for x in (0, shape[0] - 1) for y in (0, shape[1] - 1) for z in (0, shape[2] - 1)])
    lattice = np.array([[x, y, z] for x in np.linspace(0, shape[0] - 1, 5) for y in np.linspace(0, shape[1] - 1, 5) for z in np.linspace(0, shape[2] - 1, 5)])
    points = origin @ np.c_[np.r_[corners, lattice], np.ones(133)].T
    (records, gates) = ({}, [])
    previous = {'official': np.eye(4), 'fnit': np.eye(4)}
    for mode in ('rigid', 'affine'):
        (entries, warped) = ({}, {})
        for (name, folder) in (('official', old), ('fnit', new)):
            (lta_path, image_path) = (folder / (mode + '.lta'), folder / (mode + '.header.mgz'))
            (transform, image) = (load_lta(lta_path), nib.load(image_path))
            if transform.space != 'world':
                raise ValueError('expected RAS-world LTA')
            matrix = transform.matrix
            combined = native_matmul(matrix, previous[name]).astype(float)
            previous[name] = combined
            exact = tuple(image.shape) == tuple(original.shape) and np.array_equal(np.asanyarray(image.dataobj), values)
            exact = exact and image.header.get_data_dtype() == original.header.get_data_dtype()
            expected_source = original if mode == 'rigid' else nib.load(folder / 'rigid.header.mgz')
            shape_ok = transform.source.shape == tuple(expected_source.shape) and transform.target.shape == tuple(target.shape)
            ideal = native_matmul(matrix, header_affine(expected_source, legacy))
            internal = float(np.max(np.abs(header_affine(image, legacy).astype(float) - ideal.astype(float))))
            pull = native_matmul(native_inverse(header_affine(image, legacy)), header_affine(target, legacy))
            warped[name] = resample(torch.from_numpy(values.astype(np.float32)), target.shape, pull, chunk_size=plan['parameters']['spatial_chunk_size'])
            entries[name] = {'LTA_sha256': digest(lta_path), 'mapped_MGZ_sha256': digest(image_path), 'source_voxels_shape_dtype_exact': bool(exact), 'LTA_source_target_shape_exact': bool(shape_ok), 'mapped_voxel_to_RAS_internal_max_mm': internal, 'RAS_matrix': matrix.tolist(), 'combined_RAS_matrix': combined.tolist(), 'stored_delta': image.header['delta'].tolist(), 'stored_Mdc': image.header['Mdc'].tolist(), 'stored_Pxyz_c': image.header['Pxyz_c'].tolist()}
            gates.extend([bool(exact), bool(shape_ok), internal <= plan['gates']['mapped_geometry_consistency_max_mm']])
        pold = np.asarray(entries['official']['combined_RAS_matrix']) @ points
        pnew = np.asarray(entries['fnit']['combined_RAS_matrix']) @ points
        displacement = np.linalg.norm((pnew - pold)[:3], axis=0)
        diff = (warped['fnit'] - warped['official']).double()
        reference = warped['official'].double()
        if float(torch.linalg.vector_norm(reference)) == 0:
            raise ValueError('official saved geometry warps to an empty target support')
        rel = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference))
        support = int(torch.count_nonzero((warped['fnit'] != 0) ^ (warped['official'] != 0)))
        rms = float(np.sqrt(np.mean(displacement ** 2)))
        maximum = float(displacement.max())
        gates.extend([rms <= plan['gates']['physical_displacement_RMS_mm_max'], maximum <= plan['gates']['physical_displacement_max_mm_max'], rel <= plan['gates']['warped_same_target_grid_relative_L2_max'], support == 0])
        header_fields = {}
        old_header = nib.load(old / (mode + '.header.mgz')).header
        new_header = nib.load(new / (mode + '.header.mgz')).header
        for field in ('version', 'dims', 'type', 'dof', 'goodRASFlag', 'delta', 'Mdc', 'Pxyz_c', 'tr', 'flip_angle', 'te', 'ti', 'fov'):
            (a, b) = (np.asarray(old_header[field]), np.asarray(new_header[field]))
            header_fields[field] = {'exact': bool(np.array_equal(a, b)), 'max_absolute': float(np.max(np.abs(a.astype(float) - b.astype(float))))}
        records[mode] = {'outputs': entries, 'point_count': 133, 'displacement_RMS_mm': rms, 'displacement_max_mm': maximum, 'warp_relative_L2': rel, 'warp_max_absolute': float(diff.abs().max()), 'warp_P99_absolute': float(torch.quantile(diff.abs().flatten(), 0.99)), 'warp_nonzero_support_difference_voxels': support, 'world_LTA_matrix_max_absolute': float(np.max(np.abs(np.asarray(entries['fnit']['RAS_matrix']) - np.asarray(entries['official']['RAS_matrix'])))), 'combined_world_matrix_max_absolute': float(np.max(np.abs(np.asarray(entries['fnit']['combined_RAS_matrix']) - np.asarray(entries['official']['combined_RAS_matrix'])))), 'mapped_MGH_13_fields': header_fields, 'fixed_actual_mask_overlap': {'official': overlap(warped['official']), 'fnit': overlap(warped['fnit'])}, 'warp_scope': 'shared FNIT sampler on official/new saved mapped geometry; not an independent official resampler oracle'}
    return {'all_declared_gates_pass': bool(all(gates)), 'checks': len(gates), 'stages': records, 'initial_header_fixed_mask_overlap': initial_overlap, 'native_GEMS_or_optimizer_called': False, 'MRI_output_arrays_published': False}
