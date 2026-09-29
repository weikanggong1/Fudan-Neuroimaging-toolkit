"""Scratch replacements for the two external commands used by subregions."""
import shlex
import nibabel as nib
import numpy as np
from fnit.gems.native_samseg.gems import gems_resample


def _multiply_float32(left, right):
    output = np.zeros((left.shape[0], right.shape[1]), dtype=np.float32)
    for row in range(left.shape[0]):
        for column in range(right.shape[1]):
            value = np.float32(0)
            for index in range(left.shape[1]):
                value = np.float32(value + np.float32(left[row, index] * right[index, column]))
            output[row, column] = value
    return output


def _resampled_header(image, header, shape, mapping):
    source_matrix = np.eye(4, dtype=np.float32)
    source_matrix[:3, :3] = np.asarray(image.header['Mdc'], np.float32).T * np.asarray(image.header['delta'], np.float32)
    for row in range(3):
        offset = np.float64(0)
        for column in range(3):
            offset += np.float64(source_matrix[row, column]) * np.float64(np.float32(image.shape[column] / 2))
        source_matrix[row, 3] = np.float32(np.float64(image.header['Pxyz_c'][row]) - np.float64(np.float32(offset)))
    center = np.asarray([*np.asarray(shape, dtype=np.float32) / 2, 1], dtype=np.float32)[:, None]
    mapped = _multiply_float32(source_matrix, mapping)
    header['Mdc'] = (mapped[:3, :3] / np.asarray(header['delta'], np.float32)).T
    header['Pxyz_c'] = _multiply_float32(source_matrix, _multiply_float32(mapping, center))[:3, 0]
    return header


def _cubic_resample(image, reference, resolution=None):
    """Resample an MGH volume with the bundled FreeSurfer-compatible B-spline."""
    if reference is None:
        zoom = np.full(3, resolution, dtype=np.float32)
        previous_zoom = np.asarray(image.header['delta'], dtype=np.float32)
        shape = tuple(np.ceil(np.asarray(image.shape[:3]) * previous_zoom / zoom - 1e-5).astype(int))
        header = image.header.copy()
        header.set_data_shape(shape)
        header['delta'] = zoom
    else:
        shape = reference.shape[:3]
        header = reference.header.copy()
        zoom = np.asarray(header['delta'], dtype=np.float32)
    header.set_data_dtype(np.float32)
    data, mapping = gems_resample.resample_cubic(
        np.asarray(image.dataobj, dtype=np.float32),
        np.asarray(image.header['delta'], dtype=np.float32),
        np.asarray(image.header['Mdc'], dtype=np.float32),
        np.asarray(image.header['Pxyz_c'], dtype=np.float32),
        shape, zoom,
        np.asarray(header['Mdc'], dtype=np.float32),
        np.asarray(header['Pxyz_c'], dtype=np.float32),
    )
    header = _resampled_header(image, header, shape, mapping)
    output = nib.MGHImage(data, None, header=header)
    output._affine = header.get_vox2ras()
    return output


def _with_vox2ras(image, vox2ras):
    """Write a changed MGH vox2ras while retaining the original voxel sizes."""
    header = image.header.copy()
    zoom = np.asarray(header['delta'], dtype=np.float32)
    header['Mdc'] = (vox2ras[:3, :3] / zoom).T.astype(np.float32)
    center = np.asarray(image.shape[:3], dtype=np.float32) / 2
    header['Pxyz_c'] = (vox2ras[:3, 3] + vox2ras[:3, :3] @ center).astype(np.float32)
    output = nib.MGHImage(np.asarray(image.dataobj, dtype=np.float32), None, header=header)
    output._affine = header.get_vox2ras()
    return output


def run(command):
    args = shlex.split(command)
    if args[0] == 'mri_convert':
        source, target = args[1:3]
        image = nib.load(source)
        method = args[args.index('-rt') + 1]
        if method == 'cubic':
            reference = None if '-vs' in args else nib.load(args[args.index('-rl') + 1])
            resolution = float(args[args.index('-vs') + 1]) if '-vs' in args else None
            nib.save(_cubic_resample(image, reference, resolution), target)
            return
        if '-rl' not in args or method not in ('nearest', 'interpolate'):
            raise ValueError('仅支持 -rl 最近邻/三线性插值和三次 B 样条重采样')
        reference = nib.load(args[args.index('-rl') + 1])
        field = lambda volume, name: np.asarray(volume.header[name], dtype=np.float32)
        data, mapping = gems_resample.resample_simple(
            np.asarray(image.dataobj, dtype=np.float32),
            field(image, 'delta'), field(image, 'Mdc'), field(image, 'Pxyz_c'),
            reference.shape[:3], field(reference, 'delta'),
            field(reference, 'Mdc'), field(reference, 'Pxyz_c'),
            method == 'nearest',
        )
        header = reference.header.copy()
        header.set_data_dtype(np.float32)
        header = _resampled_header(image, header, reference.shape[:3], mapping)
        nib.save(nib.MGHImage(data, None, header=header), target)
        return
    if args[0] == 'mri_robust_register':
        from .native_registration import register_atlas

        mov_path = args[args.index('--mov') + 1]
        dst_path = args[args.index('--dst') + 1]
        mov, dst = nib.load(mov_path), nib.load(dst_path)
        nib.save(register_atlas(mov, dst, affine='--affine' in args), mov_path)
        return
    raise RuntimeError('Unsupported external command: ' + args[0])
