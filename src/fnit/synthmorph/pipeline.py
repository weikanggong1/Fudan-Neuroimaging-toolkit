"""SynthMorph registration with PyTorch inference and native image geometry."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
from .spatial import compose, transform
from .affine_no_surfa import AffineTransform, load_affine_image, network_space_affine
from ..weights import resolve_weights


@dataclass
class RegistrationResult:
    moved: object
    fixed_moved: object
    transform: object
    inverse: object


def network_space(image, shape, center=None):
    import surfa as sf
    new = sf.ImageGeometry(shape=shape, voxsize=1, rotation='LIA',
                           center=image.geom.center if center is None else center.geom.center,
                           shear=None)
    return ((image.geom.world2vox @ new.vox2world).matrix,
            (new.world2vox @ image.geom.vox2world).matrix)


def _load(image, single_frame=True):
    import surfa as sf
    out = sf.load_volume(str(image)) if isinstance(image, (str, Path)) else image
    if not isinstance(out, sf.Volume) or len(out.shape) not in (3, 4):
        raise ValueError('input must be a 3D volume with optional frames')
    if single_frame and len(out.shape) != 3:
        raise ValueError('registration inputs must be single-frame 3D volumes')
    if not np.isfinite(out.data).all():
        raise ValueError('input contains NaN or infinity')
    return out


def _tensor(image, device):
    return torch.as_tensor(np.array(image.data, dtype=np.float32, copy=True), device=device)[None, None]


def _numpy(tensor):
    if tensor.ndim == 2:
        return tensor.detach().cpu().numpy()
    return tensor[0].permute(1, 2, 3, 0).detach().cpu().numpy()


class SynthMorph:
    """Reusable registration model; weights is a directory or named path mapping.

    model: joint (affine + deformable), deform, affine, or rigid.
    hyper: warp regularity in (0,1), fixed for this model instance.
    extent: 192 or 256 isotropic 1-mm network voxels per axis.
    Networks and coordinate composition run on device. Final image resampling
    uses Surfa on CPU to preserve the original command's boundary conventions.
    """
    def __init__(self, weights=None, device='cpu', model='joint', extent=256,
                 hyper=0.5, steps=7):
        from .models import SynthMorphNetwork
        if model not in ('joint', 'deform', 'affine', 'rigid'):
            raise ValueError('unknown registration model')
        if extent not in (192, 256):
            raise ValueError('extent must be 192 or 256')
        if not 0 < hyper < 1 or steps < 5:
            raise ValueError('hyper must be in (0,1) and steps must be >=5')
        names = {'affine': 'synthmorph.affine.2.h5', 'deform': 'synthmorph.deform.3.h5',
                 'rigid': 'synthmorph.rigid.1.h5'}
        needed = ('affine', 'deform') if model == 'joint' else (model,)
        paths = {key: resolve_weights(names[key], weights.get(key) if isinstance(weights, dict) else weights)
                 for key in needed}
        self.device, self.model, self.extent = torch.device(device), model, extent
        # Keep FP32 tensors while allowing TF32 kernels on Ampere/Hopper.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.network = SynthMorphNetwork(weights=paths, model=model, hyper=hyper,
                                        int_steps=steps, device=device)

    @torch.inference_mode()
    def affine_transform(self, moving: str | Path, fixed: str | Path) -> AffineTransform:
        """Return the affine-only world transform without Surfa image resampling."""
        if self.model != "affine":
            raise ValueError("affine_transform requires model='affine'")
        moving_data, moving_geometry = load_affine_image(moving)
        fixed_data, fixed_geometry = load_affine_image(fixed)
        shape = (self.extent,) * 3
        net_to_moving, moving_to_net = network_space_affine(moving_geometry, shape)
        net_to_fixed, fixed_to_net = network_space_affine(fixed_geometry, shape)
        inputs = []
        for data, matrix in ((moving_data, net_to_moving), (fixed_data, net_to_fixed)):
            native = torch.as_tensor(np.array(data, dtype=np.float32, copy=True),
                                     device=self.device)[None, None]
            image = transform(native, matrix, shape=shape)
            image -= image.min()
            maximum = image.max()
            if maximum <= 0:
                raise ValueError("input has no intensity variation in network space")
            inputs.append(image / maximum)
        _, backward = self.network(*inputs)
        voxel = _numpy(compose((net_to_fixed, backward, moving_to_net),
                               shape=moving_geometry.shape))
        world = fixed_geometry.matrix @ np.asarray(voxel, np.float64) @ np.linalg.inv(moving_geometry.matrix)
        return AffineTransform(world, moving_geometry, fixed_geometry)

    @torch.inference_mode()
    def __call__(self, moving, fixed, init=None, mid_space=False, header_only=False,
                 output_dir=None):
        import surfa as sf
        mov, fix = _load(moving), _load(fixed)
        is_matrix = self.model in ('affine', 'rigid')
        if header_only and not is_matrix:
            raise ValueError('header_only requires affine or rigid model')
        if mid_space and init is None:
            raise ValueError('mid_space initialization requires init')
        shape = (self.extent,) * 3
        net_to_mov, mov_to_net = network_space(mov, shape, fix if self.model == 'deform' else None)
        net_to_fix, fix_to_net = network_space(fix, shape)
        if init is not None:
            initial = sf.load_affine(str(init)) if isinstance(init, (str, Path)) else init
            initial = initial.convert(space='voxel')
            if not sf.transform.image_geometry_equal(mov.geom, initial.source, tol=1e-3) or not sf.transform.image_geometry_equal(fix.geom, initial.target, tol=1e-3):
                raise ValueError('initial transform geometry does not match input images')
            initial = fix_to_net @ initial.matrix @ net_to_mov
            if mid_space:
                initial = scipy.linalg.sqrtm(initial)
                if np.iscomplexobj(initial) and np.max(np.abs(initial.imag)) > 1e-5:
                    raise ValueError('initial affine has no usable real square root')
                initial = np.real(initial)
                net_to_fix = net_to_fix @ initial
                fix_to_net = np.linalg.inv(net_to_fix)
            net_to_mov = net_to_mov @ np.linalg.inv(initial)
            mov_to_net = np.linalg.inv(net_to_mov)
        native = [_tensor(im, self.device) for im in (mov, fix)]
        inputs = []
        for image, matrix in zip(native, (net_to_mov, net_to_fix)):
            out = transform(image, matrix, shape=shape)
            out -= out.min()
            maximum = out.max()
            if maximum <= 0:
                raise ValueError('input has no intensity variation in network space')
            inputs.append(out / maximum)
        fw_net, bw_net = self.network(*inputs)
        fw = compose((net_to_mov, fw_net, fix_to_net), shape=fix.shape)
        bw = compose((net_to_fix, bw_net, mov_to_net), shape=mov.shape)
        if is_matrix:
            forward = sf.Affine(_numpy(bw), source=mov, target=fix, space='voxel')
            inverse = sf.Affine(_numpy(fw), source=fix, target=mov, space='voxel')
            output_format = dict(space='world')
        else:
            forward = sf.Warp(_numpy(fw), source=mov, target=fix, format=sf.Warp.Format.disp_crs)
            inverse = sf.Warp(_numpy(bw), source=fix, target=mov, format=sf.Warp.Format.disp_crs)
            output_format = dict(format=sf.Warp.Format.disp_ras)
        # Surfa has a different valid interpolation domain from the network's
        # TensorFlow sampler. Preserve native CRS transforms until after resampling.
        moved = mov.transform(forward, resample=not header_only)
        fixed_moved = fix.transform(inverse, resample=not header_only)
        forward = forward.convert(**output_format)
        inverse = inverse.convert(**output_format)
        if output_dir:
            root = Path(output_dir)
            root.mkdir(parents=True, exist_ok=True)
            net_mov = sf.Volume(_numpy(inputs[0])[..., 0], geometry=sf.ImageGeometry(shape, vox2world=mov.geom.vox2world.matrix @ net_to_mov))
            net_fix = sf.Volume(_numpy(inputs[1])[..., 0], geometry=sf.ImageGeometry(shape, vox2world=fix.geom.vox2world.matrix @ net_to_fix))
            net_mov.save(root / 'inp_1.nii.gz')
            net_fix.save(root / 'inp_2.nii.gz')
            np.savez_compressed(root / 'network_transforms.npz', forward=_numpy(fw_net), inverse=_numpy(bw_net))
        return RegistrationResult(moved, fixed_moved, forward, inverse)



@torch.inference_mode()
def apply_transform(image, transformation, method='linear', fill=0,
                    dtype='float32', header_only=False):
    """Apply an LTA or RAS warp with the original Surfa CPU resampling rules.
    """
    import surfa as sf
    image = _load(image, single_frame=False)
    if isinstance(transformation, (str, Path)):
        path = str(transformation)
        transformation = sf.load_affine(path) if path.endswith('.lta') else sf.load_warp(path)
    if header_only and not isinstance(transformation, sf.Affine):
        raise ValueError('header_only requires an affine')
    if isinstance(transformation, sf.Warp):
        # Warp source geometry determines CRS coordinates; disallow silent mismatch.
        if not sf.transform.image_geometry_equal(image.geom, transformation.source, tol=1e-3):
            raise ValueError('warp source geometry does not match image')
    return image.transform(transformation, method=method, fill=fill,
                           resample=not header_only).astype(dtype)
