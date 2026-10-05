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
from .._world_resampling import WorldTransformChain, resample_world_image
from .spatial import (
    _prepare_transform, _sample_prepared, compose, surfa_nearest, transform,
)


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


def _load(image, single_frame=True, check_finite=True):
    out = load_image(image, "input")
    if len(out.shape) not in (3, 4):
        raise ValueError("input must be a 3D volume with optional frames")
    if single_frame and len(out.shape) != 3:
        raise ValueError("registration inputs must be single-frame 3D volumes")
    if check_finite and not np.isfinite(np.asanyarray(out.dataobj)).all():
        raise ValueError("input contains NaN or infinity")
    return out


def _image_data(image, device):
    """Decode CPU images before casting, preserving NIfTI scaling precision.

    ArrayProxy can perform slope/intercept arithmetic in the requested output
    dtype. The original reader and a fully materialized SpatialImage first
    decode in the proxy's default dtype. Keep that order on CPU so both public
    input routes use the same voxel values. CUDA retains its validated decode.
    """
    data = (np.asanyarray(image.dataobj)
            if torch.device(device).type == "cpu" else image.dataobj)
    return np.array(data, dtype=np.float32, copy=True)


def _tensor(image, device):
    data = _image_data(image, device)
    return torch.as_tensor(data, device=device)[None, None]


def _tensor_frames(image, device):
    data = _image_data(image, device)
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


def _header_transform(image, transformation, data=None):
    world = transformation.convert(space="world", source=image).matrix
    return new_image(
        np.array(image.dataobj if data is None else data, copy=True),
        image, affine=world @ image.affine,
    )


def _network_input_images(inputs, moving, fixed, net_to_moving, net_to_fixed,
                          device, *, has_init=False):
    """Return the documented debug inputs in their preview geometries."""
    moving_affine = moving.affine @ net_to_moving
    if torch.device(device).type == "cpu" and has_init:
        # Initial alignment moves voxel data into the fixed network grid.
        # The original debug preview labels both inputs with that geometry.
        moving_affine = fixed.affine @ net_to_fixed
    return (
        new_image(_numpy(inputs[0])[..., 0], moving, affine=moving_affine),
        new_image(_numpy(inputs[1])[..., 0], fixed,
                  affine=fixed.affine @ net_to_fixed),
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
    elif torch.device(device).type == 'cpu' and method == 'linear':
        # Surfa's final image sampler accepts the last-center band [n-1,n).
        # Network preprocessing/integration keep their Neurite rules; this
        # CPU-only image boundary correction leaves CUDA outputs unchanged.
        plan = _prepare_transform(
            pull, tensor.shape[2:], device=tensor.device, dtype=tensor.dtype,
            shape=image_geometry(target).shape, method=method, fill_value=fill,
            surfa_linear_rule=True,
        )
        moved = _sample_prepared(tensor, plan, fill)
    else:
        moved = transform(
            tensor,
            pull,
            shape=image_geometry(target).shape,
            method=method,
            fill_value=fill,
        )
    return new_image(_tensor_data(moved), image, affine=image_geometry(target).affine)


def _resampled_frames(image, data, pull, target, device, *, method, fill, frame_chunk_size):
    """Decode once and sample frames with one reusable coordinate plan.

    Default CPU chunks contain at most 32 frames. CUDA chooses changing
    frame buffers below 8 GiB and the remaining 20 GB allocation budget,
    measured after coordinate transfer with 512 MiB held in reserve.
    """
    device = torch.device(device)
    geometry = image_geometry(target)
    frames = data[None] if data.ndim == 3 else np.moveaxis(data, -1, 0)
    count = frames.shape[0]
    if frame_chunk_size is not None and (
        isinstance(frame_chunk_size, (bool, np.bool_))
        or not isinstance(frame_chunk_size, (int, np.integer))
        or frame_chunk_size < 1
    ):
        raise ValueError("frame_chunk_size must be a positive integer or None")

    plan = _prepare_transform(
        pull, data.shape[:3], device="cpu", dtype=torch.float32,
        shape=geometry.shape, method=method, fill_value=fill,
        surfa_nearest_rule=True, surfa_linear_rule=device.type == 'cpu',
        surfa_nearest_half_up=device.type == 'cpu',
    )
    # Independent apply keeps the original CPU float32 coordinate arithmetic.
    # Transferring a prepared plan avoids TF32 affine-coordinate drift while
    # GPU interpolation still processes every frame on the selected device.
    if device.type != "cpu":
        plan = plan.to(device)
    if device.type == "cuda":
        # Include the prepared coordinates and existing device allocations.
        # Reserve 512 MiB below 20 GB for sampler/runtime working storage.
        available = (
            20_000_000_000 - torch.cuda.memory_allocated(device) - 512 * 1024 ** 2
        )
        frame_bytes = 4 * (
            int(np.prod(data.shape[:3])) + 2 * int(np.prod(geometry.shape))
        )
        if available < frame_bytes:
            raise RuntimeError(
                "CUDA frame buffers cannot fit one frame within the 20 GB "
                f"memory budget: {frame_bytes} bytes needed, {available} available"
            )
        if frame_chunk_size is None:
            buffer_budget = min(8 * 1024 ** 3, available)
            if buffer_budget < frame_bytes:
                raise RuntimeError(
                    "automatic CUDA frame buffer budget cannot fit one frame: "
                    f"{frame_bytes} bytes needed, {buffer_budget} available"
                )
            frame_chunk_size = min(count, buffer_budget // frame_bytes)
        else:
            frame_chunk_size = min(count, frame_chunk_size)
            if frame_chunk_size * frame_bytes > available:
                raise RuntimeError(
                    "frame_chunk_size exceeds the CUDA 20 GB memory budget: "
                    f"{frame_chunk_size * frame_bytes} bytes needed, {available} available"
                )
    else:
        frame_chunk_size = (
            min(count, 32) if frame_chunk_size is None else min(count, frame_chunk_size)
        )
    # Keep the returned array on the host while only a bounded frame chunk is
    # resident on the GPU. Moving frames does not change interpolation math.
    output = None
    for start in range(0, count, frame_chunk_size):
        stop = min(start + frame_chunk_size, count)
        chunk = np.asarray(frames[start:stop], dtype=np.float32)
        # The sampler only reads its input. Reuse writable float32 decoded
        # storage, but copy arrays that torch cannot safely expose as a view.
        if not chunk.flags.writeable or any(stride < 0 for stride in chunk.strides):
            chunk = np.array(chunk, copy=True)
        tensor = torch.as_tensor(chunk, device=device)[None]
        moved = _sample_prepared(tensor, plan, fill)
        sampled = moved[0].cpu().numpy()
        if start == 0 and stop == count:
            # Preserve the one-shot CPU path's zero-copy tensor result view.
            output = sampled
        else:
            if output is None:
                output = np.empty((count, *geometry.shape), dtype=np.float32)
            output[start:stop] = sampled
        del tensor, moved
    # Retain a singleton frame axis for a 4D input. Registration's separate
    # single-frame 3D path intentionally continues to return a 3D volume.
    output = output[0] if data.ndim == 3 else np.moveaxis(output, 0, -1)
    return new_image(output, image, affine=geometry.affine)


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
        if configure_precision and self.device.type == "cuda":
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
        if self.device.type == 'cpu' and self.model == 'joint':
            from ._cpu_preprocessing import network_transform
            input_sampler = network_transform
        else:
            input_sampler = transform
        inputs = []
        for image, matrix in zip(native, (net_to_mov, net_to_fix)):
            normalized = input_sampler(image, matrix, shape=shape)
            normalized -= normalized.min()
            maximum = normalized.max()
            if maximum <= 0:
                raise ValueError("input has no intensity variation in network space")
            inputs.append(normalized / maximum)

        if precision_report is not None:
            from fnit.recon_all.profiling import record_network_forward
            record_network_forward(self.network, inputs[0], precision_report,
                                   model=self.model, compute_inverse=bool(compute_inverse))
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
            if self.device.type == "cpu" and self.model == "rigid":
                # Associate each returned rigid affine with the exact inverse
                # of its own image pull. Independently rounded FP32 reciprocal
                # predictions need not be exact inverses after composition.
                # Invert the already-associated world matrices in float64 so
                # final sampling and public reapplication consume one map.
                forward, inverse = (
                    AffineTransform(np.linalg.inv(inverse.matrix),
                                    source=mov, target=fix, space="world"),
                    AffineTransform(np.linalg.inv(forward.matrix),
                                    source=fix, target=mov, space="world"),
                )
            if transform_only:
                moved = fixed_moved = None
            elif header_only:
                moved = _header_transform(mov, forward)
                fixed_moved = _header_transform(fix, inverse)
            elif self.device.type == "cpu":
                # The original final affine sampler inverts the associated
                # returned affine. The network's two float32 predictions are
                # only approximately reciprocal; sampling with the opposite
                # prediction can cross a fill boundary. Use the public affine
                # application route for both CPU images and preserve CUDA's
                # previously validated paired-prediction sampling below.
                moved = apply_transform(mov, forward, device=self.device)
                fixed_moved = apply_transform(fix, inverse, device=self.device)
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
            net_mov, net_fix = _network_input_images(
                inputs, mov, fix, net_to_mov, net_to_fix, self.device,
                has_init=init is not None,
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
    *,
    device="cpu",
    frame_chunk_size=None,
    boundary="grid-constant",
    output_mask=None,
    spatial_chunk_size=262144,
):
    """Apply an affine, RAS displacement or world transform chain to 3D/4D data.

    ``device='cpu'`` preserves the original default sampler. CUDA is opt-in;
    floating-point interpolation can differ from CPU by rounding. A positive
    ``frame_chunk_size`` bounds frame buffers and reuses the same coordinates.
    None selects at most 32 frames on CPU. CUDA uses at most 8 GiB of changing
    frame buffers within 20 GB minus current allocated memory and 512 MiB.
    The budget counts one input and two float32 output buffers per frame;
    it is checked after the coordinate plan is resident on CUDA. Explicit
    CUDA chunks use the same remaining 20 GB budget. A one-shot CPU result
    retains its zero-copy output view (by default when there are <=32 frames).
    Coordinates use the original CPU float32 arithmetic and are transferred
    once for CUDA sampling. Image decoding and NIfTI I/O remain on the CPU.

    A ``WorldTransformChain`` uses the shared volume sampler, including cubic
    ``method='spline'``, its ``boundary``, output mask and spatial chunk size.
    It composes the fixed-grid RAS pull, world affine and per-frame motion in
    one interpolation. Its frame chunk is the sampler batch size (None: 8).
    This branch requires zero fill and cannot perform header-only updates.
    Default float32 returns the shared image unchanged, including its header.
    """
    if isinstance(transformation, WorldTransformChain):
        if header_only:
            raise ValueError("header_only is unsupported for WorldTransformChain")
        if fill is None or fill != 0:
            raise ValueError("WorldTransformChain requires fill=0")
        if method not in ("linear", "nearest", "spline"):
            raise ValueError(
                "WorldTransformChain method must be linear, nearest or spline"
            )
        if frame_chunk_size is not None and (
            isinstance(frame_chunk_size, (bool, np.bool_))
            or not isinstance(frame_chunk_size, (int, np.integer))
            or frame_chunk_size < 1
        ):
            raise ValueError("frame_chunk_size must be a positive integer or None")
        output_dtype = np.dtype(dtype)
        result = resample_world_image(
            image,
            transformation.reference,
            transformation.reference_to_source_world,
            pre_affine_pull_ras=transformation.pre_affine_pull_ras,
            output_mask=output_mask,
            interpolation=method,
            boundary=boundary,
            motion_pull_world=transformation.motion_pull_world,
            coordinate_precision=transformation.coordinate_precision,
            spatial_chunk_size=spatial_chunk_size,
            batch_size=8 if frame_chunk_size is None else int(frame_chunk_size),
            device=device,
        )
        if output_dtype == np.dtype(np.float32):
            # Preserve the mature volume sampler's complete image contract.
            return result
        header = result.header.copy()
        header.set_data_dtype(output_dtype)
        return nib.Nifti1Image(
            np.asarray(result.dataobj).astype(output_dtype, copy=False),
            result.affine,
            header,
        )
    if (
        boundary != "grid-constant" or output_mask is not None
        or spatial_chunk_size != 262144
    ):
        raise ValueError(
            "boundary, output_mask and spatial_chunk_size require WorldTransformChain"
        )
    image = _load(image, single_frame=False, check_finite=False)
    # Materialise an ArrayProxy only once (especially important for .nii.gz).
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError("input contains NaN or infinity")
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
        result = _header_transform(image, transformation, data=data)
    elif isinstance(transformation, AffineTransform):
        world = transformation.convert(space="world").matrix
        target = transformation.target
        pull = (
            np.linalg.inv(image.affine)
            @ np.linalg.inv(world)
            @ target.affine
        )
        result = _resampled_frames(
            image,
            data,
            pull,
            target,
            device,
            method=method,
            fill=fill,
            frame_chunk_size=frame_chunk_size,
        )
    elif isinstance(transformation, DenseWarp):
        if not same_geometry(image, transformation.source):
            raise ValueError("warp source geometry does not match image")
        pull = ras_displacement_to_voxel(
            np.asanyarray(transformation.dataobj), image, transformation.target
        )
        pull = torch.as_tensor(pull).permute(3, 0, 1, 2)[None]
        result = _resampled_frames(
            image,
            data,
            pull,
            transformation.target,
            device,
            method=method,
            fill=fill,
            frame_chunk_size=frame_chunk_size,
        )
    else:
        raise TypeError("transformation must be an LTA affine or dense warp")

    output_dtype = np.dtype(dtype)
    return new_image(
        np.asarray(result.dataobj).astype(output_dtype, copy=False), result
    )
