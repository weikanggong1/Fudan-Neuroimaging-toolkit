"""SynthMorph registration with PyTorch inference and nibabel image I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.linalg
import torch

from .._nib import FNITNifti1Image, load_image, new_image
from .._transforms import (
    AffineTransform,
    DenseWarp,
    image_geometry,
    load_dense_warp,
    load_lta,
    ras_displacement_to_voxel,
    same_geometry,
    voxel_displacement_to_ras,
)
from ..weights import resolve_weights
from .spatial import compose, surfa_nearest, transform


@dataclass
class RegistrationResult:
    moved: FNITNifti1Image | None
    fixed_moved: FNITNifti1Image | None
    transform: AffineTransform | DenseWarp
    inverse: AffineTransform | DenseWarp | None


def network_space(image, shape, center=None):
    """Return network-to-image and image-to-network voxel transforms.

    The network grid is 1 mm LIA and uses FreeSurfer's ``shape / 2`` centre
    convention. ``center`` can select another image's world-space centre.
    """
    image = image_geometry(image)
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("network shape must have three positive dimensions")
    world_center = image.center if center is None else image_geometry(center).center
    rotation = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    network_affine = np.eye(4, dtype=np.float64)
    network_affine[:3, :3] = rotation
    network_affine[:3, 3] = world_center - rotation @ (
        np.asarray(shape, dtype=np.float64) / 2.0
    )
    network_to_image = np.linalg.inv(image.affine) @ network_affine
    image_to_network = np.linalg.inv(network_affine) @ image.affine
    return network_to_image, image_to_network


def _load(image, single_frame=True):
    out = load_image(image, "input")
    if len(out.shape) not in (3, 4):
        raise ValueError("input must be a 3D volume with optional frames")
    if single_frame and len(out.shape) != 3:
        raise ValueError("registration inputs must be single-frame 3D volumes")
    if not np.isfinite(np.asanyarray(out.dataobj)).all():
        raise ValueError("input contains NaN or infinity")
    return out


def _tensor(image, device):
    data = np.array(image.dataobj, dtype=np.float32, copy=True)
    return torch.as_tensor(data, device=device)[None, None]


def _tensor_frames(image, device):
    data = np.array(image.dataobj, dtype=np.float32, copy=True)
    if data.ndim == 3:
        return torch.as_tensor(data, device=device)[None, None]
    data = np.moveaxis(data, -1, 0)
    return torch.as_tensor(data, device=device)[None]


def _numpy(tensor):
    if tensor.ndim == 2:
        return tensor.detach().cpu().numpy()
    return tensor[0].permute(1, 2, 3, 0).detach().cpu().numpy()


def _tensor_data(tensor):
    data = tensor[0].detach().cpu().numpy()
    if data.shape[0] == 1:
        return data[0]
    return np.moveaxis(data, 0, -1)


def _affine_input(value, moving, fixed):
    if isinstance(value, (str, Path)):
        value = load_lta(value)
    if isinstance(value, AffineTransform):
        if not same_geometry(moving, value.source) or not same_geometry(
            fixed, value.target
        ):
            raise ValueError("initial transform geometry does not match input images")
        return value.convert(space="voxel", source=moving, target=fixed)
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("init must be an LTA path or finite 4x4 affine") from error
    return AffineTransform(
        matrix, source=moving, target=fixed, space="world"
    ).convert(space="voxel")


def _header_transform(image, transformation):
    world = transformation.convert(space="world", source=image).matrix
    return new_image(
        np.array(image.dataobj, copy=True), image, affine=world @ image.affine
    )


def _resampled_image(
    image,
    pull,
    target,
    device,
    *,
    method="linear",
    fill=0,
    surfa_nearest_rule=False,
):
    tensor = _tensor_frames(image, device)
    if method == "nearest" and surfa_nearest_rule:
        moved = surfa_nearest(
            tensor,
            pull,
            shape=image_geometry(target).shape,
            fill_value=fill,
        )
    else:
        moved = transform(
            tensor,
            pull,
            shape=image_geometry(target).shape,
            method=method,
            fill_value=fill,
        )
    return new_image(_tensor_data(moved), image, affine=image_geometry(target).affine)


class SynthMorph:
    """复用刚体、仿射和非线性模型；configure_precision=False保留调用方TF32策略。

    configure_precision默认True，与既有独立接口一致；不会启用半精度。
    recon-all构造后单独应用并记录已验证的FP32例外。
    """

    def __init__(
        self,
        weights=None,
        device="cpu",
        model="joint",
        extent=256,
        hyper=0.5,
        steps=7,
        configure_precision=True,
    ):
        from .models import SynthMorphNetwork

        if model not in ("joint", "deform", "affine", "rigid"):
            raise ValueError("unknown registration model")
        if extent not in (192, 256):
            raise ValueError("extent must be 192 or 256")
        if not 0 < hyper < 1 or steps < 5:
            raise ValueError("hyper must be in (0,1) and steps must be >=5")
        names = {
            "affine": "synthmorph.affine.2.h5",
            "deform": "synthmorph.deform.3.h5",
            "rigid": "synthmorph.rigid.1.h5",
        }
        needed = ("affine", "deform") if model == "joint" else (model,)
        paths = {
            key: resolve_weights(
                names[key], weights.get(key) if isinstance(weights, dict) else weights
            )
            for key in needed
        }
        self.device = torch.device(device)
        self.model = model
        self.extent = extent
        if configure_precision:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.network = SynthMorphNetwork(
            weights=paths,
            model=model,
            hyper=hyper,
            int_steps=steps,
            device=device,
        )

    @torch.inference_mode()
    def __call__(
        self,
        moving,
        fixed,
        init=None,
        mid_space=False,
        header_only=False,
        output_dir=None,
        transform_only=False,
        compute_inverse=True,
        precision_report=None,
    ):
        """计算 moving 到 fixed 的配准，返回带图像几何的变换。

        compute_inverse默认True保留双向输出；False仅用于非线性transform_only、
        无调试目录时，省去未消费的负velocity积分，inverse返回None。
        两次反对称网络前向保持不变。precision_report可收集实际前向精度。

        moving/fixed 是单帧 3D 图像或路径；init 是匹配这两幅图几何的
        LTA 或 4×4 世界坐标仿射。mid_space 默认关闭；header_only 默认
        关闭且仅适用于 affine/rigid；output_dir 默认不写调试图。
        transform_only 默认关闭；开启后 moved/fixed_moved 为 None，
        transform/inverse 仍保留原坐标空间和单位，省去两幅重采样图。
        无效参数或输入几何不符时抛出异常。
        """
        mov, fix = _load(moving), _load(fixed)
        is_matrix = self.model in ("affine", "rigid")
        if header_only and not is_matrix:
            raise ValueError("header_only requires affine or rigid model")
        if transform_only and header_only:
            raise ValueError("transform_only and header_only cannot be combined")
        if not compute_inverse and (is_matrix or not transform_only or output_dir):
            raise ValueError("compute_inverse=False requires nonlinear transform_only without debug output")
        if mid_space and init is None:
            raise ValueError("mid_space initialization requires init")

        shape = (self.extent,) * 3
        shared_center = fix if self.model == "deform" else None
        net_to_mov, mov_to_net = network_space(mov, shape, shared_center)
        net_to_fix, fix_to_net = network_space(fix, shape)
        if init is not None:
            initial = _affine_input(init, mov, fix)
            initial_network = fix_to_net @ initial.matrix @ net_to_mov
            if mid_space:
                initial_network = scipy.linalg.sqrtm(initial_network)
                if np.iscomplexobj(initial_network) and np.max(
                    np.abs(initial_network.imag)
                ) > 1e-5:
                    raise ValueError("initial affine has no usable real square root")
                initial_network = np.real(initial_network)
                net_to_fix = net_to_fix @ initial_network
                fix_to_net = np.linalg.inv(net_to_fix)
            net_to_mov = net_to_mov @ np.linalg.inv(initial_network)
            mov_to_net = np.linalg.inv(net_to_mov)

        native = [_tensor(image, self.device) for image in (mov, fix)]
        inputs = []
        for image, matrix in zip(native, (net_to_mov, net_to_fix)):
            normalized = transform(image, matrix, shape=shape)
            normalized -= normalized.min()
            maximum = normalized.max()
            if maximum <= 0:
                raise ValueError("input has no intensity variation in network space")
            inputs.append(normalized / maximum)

        if precision_report is not None:
            from fnit.recon_all.profiling import autocast_state
            precision_report.append({
                "model": self.model, "device": str(inputs[0].device),
                "input_dtype": str(inputs[0].dtype),
                "model_dtypes": sorted({str(p.dtype) for p in self.network.parameters()}),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "autocast": autocast_state(inputs[0].device.type),
                "compute_inverse": bool(compute_inverse),
            })
        if compute_inverse:
            forward_network, backward_network = self.network(*inputs)
        else:
            forward_network, backward_network = self.network(*inputs, compute_inverse=False)
        forward_pull = compose(
            (net_to_mov, forward_network, fix_to_net), shape=fix.shape
        )
        inverse_pull = (compose(
            (net_to_fix, backward_network, mov_to_net), shape=mov.shape
        ) if compute_inverse else None)

        if is_matrix:
            forward_voxel = _numpy(inverse_pull)
            inverse_voxel = _numpy(forward_pull)
            # float32 affine averaging/composition can perturb the fixed
            # homogeneous row even though the model estimates only 3x4 terms.
            forward_voxel[3] = (0, 0, 0, 1)
            inverse_voxel[3] = (0, 0, 0, 1)
            forward = AffineTransform(
                forward_voxel, source=mov, target=fix, space="voxel"
            ).convert(space="world")
            inverse = AffineTransform(
                inverse_voxel, source=fix, target=mov, space="voxel"
            ).convert(space="world")
            if transform_only:
                moved = fixed_moved = None
            elif header_only:
                moved = _header_transform(mov, forward)
                fixed_moved = _header_transform(fix, inverse)
            else:
                moved = _resampled_image(
                    mov, forward_pull, fix, self.device, fill=0
                )
                fixed_moved = _resampled_image(
                    fix, inverse_pull, mov, self.device, fill=0
                )
        else:
            forward = DenseWarp(
                voxel_displacement_to_ras(_numpy(forward_pull), mov, fix),
                source=mov,
                target=fix,
            )
            inverse = (DenseWarp(
                voxel_displacement_to_ras(_numpy(inverse_pull), fix, mov),
                source=fix,
                target=mov,
            ) if compute_inverse else None)
            if transform_only:
                moved = fixed_moved = None
            else:
                moved = _resampled_image(mov, forward_pull, fix, self.device, fill=0)
                fixed_moved = _resampled_image(
                    fix, inverse_pull, mov, self.device, fill=0
                )

        if output_dir:
            root = Path(output_dir)
            root.mkdir(parents=True, exist_ok=True)
            net_mov = new_image(
                _numpy(inputs[0])[..., 0], mov, affine=mov.affine @ net_to_mov
            )
            net_fix = new_image(
                _numpy(inputs[1])[..., 0], fix, affine=fix.affine @ net_to_fix
            )
            nib.save(net_mov, str(root / "inp_1.nii.gz"))
            nib.save(net_fix, str(root / "inp_2.nii.gz"))
            np.savez_compressed(
                root / "network_transforms.npz",
                forward=_numpy(forward_network),
                inverse=_numpy(backward_network),
            )
        return RegistrationResult(moved, fixed_moved, forward, inverse)


@torch.inference_mode()
def apply_transform(
    image,
    transformation,
    method="linear",
    fill=0,
    dtype="float32",
    header_only=False,
):
    """Apply an FNIT LTA affine or target-grid RAS displacement field."""
    image = _load(image, single_frame=False)
    if isinstance(transformation, (str, Path)):
        path = Path(transformation)
        transformation = (
            load_lta(path)
            if path.suffix.lower() == ".lta"
            else load_dense_warp(path, source=image)
        )
    elif isinstance(transformation, nib.spatialimages.SpatialImage) and not isinstance(
        transformation, DenseWarp
    ):
        transformation = DenseWarp(
            np.asanyarray(transformation.dataobj),
            source=image,
            target=transformation,
        )

    if header_only:
        if not isinstance(transformation, AffineTransform):
            raise ValueError("header_only requires an affine")
        result = _header_transform(image, transformation)
    elif isinstance(transformation, AffineTransform):
        world = transformation.convert(space="world").matrix
        target = transformation.target
        pull = (
            np.linalg.inv(image.affine)
            @ np.linalg.inv(world)
            @ target.affine
        )
        result = _resampled_image(
            image,
            pull,
            target,
            "cpu",
            method=method,
            fill=fill,
            surfa_nearest_rule=True,
        )
    elif isinstance(transformation, DenseWarp):
        if not same_geometry(image, transformation.source):
            raise ValueError("warp source geometry does not match image")
        pull = ras_displacement_to_voxel(
            np.asanyarray(transformation.dataobj), image, transformation.target
        )
        pull = torch.as_tensor(pull).permute(3, 0, 1, 2)[None]
        result = _resampled_image(
            image,
            pull,
            transformation.target,
            "cpu",
            method=method,
            fill=fill,
            surfa_nearest_rule=True,
        )
    else:
        raise TypeError("transformation must be an LTA affine or dense warp")

    output_dtype = np.dtype(dtype)
    return new_image(
        np.asarray(result.dataobj).astype(output_dtype, copy=False), result
    )
