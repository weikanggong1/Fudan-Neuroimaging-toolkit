"""FreeSurfer-compatible robust atlas registration without FreeSurfer binaries."""

import nibabel as nib
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation, ornt_transform
import numpy as np
from scipy.linalg import sqrtm
import torch

from fnit.gems.native_samseg.gems import gems_resample, gems_warp
from .native_free_io import _multiply_float32


def _geometry(image, name):
    return np.asarray(image.header[name], dtype=np.float32)


def _resample(image, shape, zoom):
    zoom = np.full(3, zoom, dtype=np.float32)
    header = image.header.copy()
    header.set_data_shape(shape)
    header.set_data_dtype(np.float32)
    header['delta'] = zoom
    data, _ = gems_resample.resample_cubic(
        np.asarray(image.dataobj, dtype=np.float32),
        _geometry(image, 'delta'), _geometry(image, 'Mdc'), _geometry(image, 'Pxyz_c'),
        shape, zoom, _geometry(image, 'Mdc'), _geometry(image, 'Pxyz_c'),
    )
    output = nib.MGHImage(data, None, header=header)
    output._affine = header.get_vox2ras()
    return output


def _common_grid(source, target):
    orientation = ornt_transform(io_orientation(source.affine), io_orientation(target.affine))
    data = apply_orientation(np.asarray(source.dataobj), orientation)
    permutation = np.argsort(orientation[:, 0].astype(int))
    old_zoom = _geometry(source, 'delta')
    source_matrix = np.eye(4, dtype=np.float32)
    source_matrix[:3, :3] = (_geometry(source, 'Mdc').T.astype(np.float64) * old_zoom).astype(np.float32)
    old_center = np.asarray([*np.asarray(source.shape, np.float32) / 2, 1], np.float32)[:, None]
    source_matrix[:3, 3] = _geometry(source, 'Pxyz_c') - _multiply_float32(source_matrix, old_center)[:3, 0]
    mapped = _multiply_float32(source_matrix, inv_ornt_aff(orientation, source.shape).astype(np.float32))
    header = source.header.copy()
    header.set_data_shape(data.shape)
    header['delta'] = old_zoom[permutation]
    norms = np.linalg.norm(mapped[:3, :3].astype(np.float64), axis=0)
    header['Mdc'] = (mapped[:3, :3] / norms).T.astype(np.float32)
    centered = np.eye(4, dtype=np.float32)
    centered[:3, :3] = (np.asarray(header['Mdc'], np.float64).T * np.asarray(header['delta'], np.float64)).astype(np.float32)
    centered[:3, 3] = mapped[:3, 3]
    new_center = np.asarray([*np.asarray(data.shape, np.float32) / 2, 1], np.float32)[:, None]
    header['Pxyz_c'] = _multiply_float32(centered, new_center)[:3, 0]
    reordered = nib.MGHImage(data, None, header=header)
    reordered._affine = header.get_vox2ras()
    zoom = np.float32(round(max(min(_geometry(reordered, 'delta')), min(_geometry(target, 'delta'))), 4))
    shape = tuple(np.maximum(
        np.ceil(np.asarray(reordered.shape) * _geometry(reordered, 'delta') / zoom),
        np.ceil(np.asarray(target.shape) * _geometry(target, 'delta') / zoom),
    ).astype(int))
    return _resample(reordered, shape, zoom), _resample(target, shape, zoom)


def _rigid_update(parameters):
    rotation = parameters[3:]
    angle = torch.linalg.vector_norm(rotation)
    matrix = torch.eye(4, dtype=torch.float64)
    if angle >= 1e-10:
        axis = rotation / angle
        cross = torch.zeros((3, 3), dtype=torch.float64)
        cross[0, 1], cross[0, 2] = -axis[2], axis[1]
        cross[1, 0], cross[1, 2] = axis[2], -axis[0]
        cross[2, 0], cross[2, 1] = -axis[1], axis[0]
        matrix[:3, :3] += torch.sin(angle) * cross + (1 - torch.cos(angle)) * (cross @ cross)
    matrix[:3, 3] = parameters[:3]
    return matrix


def _register(source, target, affine):
    src = np.asarray(source.dataobj, dtype=np.float32)
    dst = np.asarray(target.dataobj, dtype=np.float32)
    levels = [(src, dst)]
    while min(levels[-1][0].shape) // 2 >= 16:
        a, b = levels[-1]
        levels.append((gems_warp.pyramid_step(a), gems_warp.pyramid_step(b)))
    translation = np.asarray(gems_warp.weighted_centroid(dst)) - np.asarray(gems_warp.weighted_centroid(src))
    transform = torch.eye(4, dtype=torch.float64)
    transform[:3, 3] = torch.from_numpy(translation)
    transform[:3, 3] /= 2 ** (len(levels) - 1)
    for index in range(len(levels) - 1, -1, -1):
        if index < len(levels) - 1:
            transform[:3, 3] *= 2
        a, b = levels[index]
        for _ in range(5):
            previous = transform.clone()
            half = torch.from_numpy(np.real(sqrtm(transform.numpy())).copy())
            other_half = half @ torch.linalg.inv(transform)
            warped_a = gems_warp.warp_linear(a, half.numpy().astype(np.float32), a.shape)
            warped_b = gems_warp.warp_linear(b, other_half.numpy().astype(np.float32), a.shape)
            build = gems_warp.construct_affine if affine else gems_warp.construct_rigid
            design, residual, _ = build(warped_a, warped_b)
            if len(residual) == 0:
                raise RuntimeError('配准图像没有重叠的有效体素')
            p, _, _, _, _ = gems_warp.irls_float(design, residual, 50)
            parameters = torch.from_numpy(p).double()
            update = torch.eye(4, dtype=torch.float64)
            if affine:
                update[:3, :4] += parameters.reshape(3, 4)
            else:
                update = _rigid_update(parameters)
            transform = torch.linalg.inv(other_half) @ update @ half
            change = transform - previous
            distance = torch.sqrt((change[:3, :3] ** 2).sum() * 2000 + (change[:3, 3] ** 2).sum())
            if distance <= 0.01:
                break
    return transform.numpy()


def register_atlas(source, target, affine=False):
    """Return source MGH image with atlas-to-subject vox2ras adjusted in its header."""
    from .native_free_io import _with_vox2ras

    resampled_source, resampled_target = _common_grid(source, target)
    transform = _register(resampled_source, resampled_target, affine)
    ras_to_ras = resampled_target.affine @ transform @ np.linalg.inv(resampled_source.affine)
    return _with_vox2ras(source, ras_to_ras @ source.affine)
