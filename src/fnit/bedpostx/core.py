"""Independent PyTorch implementation of the BEDPOSTX ball-and-stick model.

The model follows Behrens et al. (2007) and the multi-shell extension of
Jbabdi et al. (2012). The MCMC implementation does not reuse FSL C++ code.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch


@dataclass(frozen=True)
class BedpostXConfig:
    nfibres: int = 3
    model: int = 2
    burnin: int = 1000
    njumps: int = 1250
    sample_every: int = 25
    ard_weight: float = 1.0
    chunk_size: int = 16384
    seed: int = 8665904

    def __post_init__(self):
        if self.nfibres not in (1, 2, 3) or self.model not in (1, 2):
            raise ValueError("nfibres must be 1, 2, or 3 and model must be 1 or 2")
        if self.burnin < 0 or self.njumps < 1 or self.sample_every < 1:
            raise ValueError("burnin, njumps, and sample_every must be valid counts")
        if self.njumps % self.sample_every:
            raise ValueError("njumps must be divisible by sample_every")
        if self.ard_weight < 0 or self.chunk_size < 1:
            raise ValueError("ard_weight and chunk_size must be nonnegative/positive")


@dataclass(frozen=True)
class BedpostXResult:
    output_dir: Path
    nvoxels: int
    nsamples: int
    elapsed_seconds: float


def _signal_model(s0, d, d_std, fractions, theta, phi, bvals, bvecs, model):
    """Return predicted DWI signal, including the isotropic compartment."""
    directions = torch.stack((torch.sin(theta) * torch.cos(phi),
                              torch.sin(theta) * torch.sin(phi),
                              torch.cos(theta)), dim=-1)
    dot2 = torch.einsum("bkc,nc->bkn", directions, bvecs).square()
    bd = bvals[None, :] * d[:, None]
    if model == 1:
        ball = torch.exp(-bd)
        sticks = torch.exp(-bd[:, None, :] * dot2)
    else:
        variance = d_std.square().clamp_min(1e-12)
        shape = d.square() / variance
        scale = variance / d.clamp_min(1e-8)
        bs = bvals[None, :] * scale[:, None]
        ball = torch.where(d_std[:, None] < 1e-5, torch.exp(-bd),
                           torch.exp(-shape[:, None] * torch.log1p(bs)))
        sticks = torch.where(d_std[:, None, None] < 1e-5,
                             torch.exp(-bd[:, None, :] * dot2),
                             torch.exp(-shape[:, None, None]
                                       * torch.log1p(bs[:, None, :] * dot2)))
    return s0[:, None] * ((1 - fractions.sum(dim=1))[:, None] * ball
                          + (fractions[:, :, None] * sticks).sum(dim=1))


def _energy(state, observed, bvals, bvecs, config):
    s0, d, d_std, fractions, theta, phi = (
        state[key] for key in ("s0", "d", "d_std", "f", "th", "ph")
    )
    prediction = _signal_model(s0.clamp_min(1e-8), d.clamp_min(1e-8),
                               d_std.clamp_min(1e-8), fractions, theta, phi,
                               bvals, bvecs, config.model)
    residual = (observed - prediction).square().sum(dim=1).clamp_min(1e-12)
    energy = (observed.shape[1] / 2) * torch.log(residual / 2)
    energy = energy - torch.log(torch.sin(theta).abs().clamp_min(1e-6)).sum(dim=1)
    if config.nfibres > 1:
        energy = energy + config.ard_weight * torch.log(
            fractions[:, 1:].clamp_min(1e-8)).sum(dim=1)
    if config.model == 2:
        energy = energy + torch.log(d_std.clamp_min(1e-8))
    valid = ((s0 > 0) & (d > 0) & (d <= 0.005)
             & (fractions > 0).all(dim=1) & (fractions < 1).all(dim=1)
             & (fractions.sum(dim=1) <= 1))
    if config.model == 2:
        valid = valid & (d_std > 0) & (d_std <= 0.01)
    return torch.where(valid & torch.isfinite(energy), energy,
                       torch.full_like(energy, float("inf")))


def _initial_state(observed, bvals, bvecs, config):
    """Initialize the chain from a log-linear diffusion-tensor estimate."""
    baseline = observed[:, bvals <= 50]
    s0 = baseline.mean(dim=1).clamp_min(1e-6)
    design = torch.stack((bvecs[:, 0].square(),
                          2 * bvecs[:, 0] * bvecs[:, 1],
                          2 * bvecs[:, 0] * bvecs[:, 2],
                          bvecs[:, 1].square(),
                          2 * bvecs[:, 1] * bvecs[:, 2],
                          bvecs[:, 2].square()), dim=1) * -bvals[:, None]
    weighted = (design.T @ design).add_(torch.eye(6, device=observed.device) * 1e-3)
    log_signal = torch.log(observed.clamp_min(1e-6) / s0[:, None])
    coef = torch.linalg.solve(weighted, design.T @ log_signal.T).T
    tensor = torch.empty((observed.shape[0], 3, 3), device=observed.device)
    tensor[:, 0, 0], tensor[:, 0, 1], tensor[:, 0, 2] = coef[:, 0], coef[:, 1], coef[:, 2]
    tensor[:, 1, 0], tensor[:, 1, 1], tensor[:, 1, 2] = coef[:, 1], coef[:, 3], coef[:, 4]
    tensor[:, 2, 0], tensor[:, 2, 1], tensor[:, 2, 2] = coef[:, 2], coef[:, 4], coef[:, 5]
    eigenvalues, eigenvectors = torch.linalg.eigh(tensor)
    d = eigenvalues[:, -1].clamp(0.0006, 0.003)
    directions = eigenvectors.flip(dims=(2,))[:, :, :config.nfibres]
    theta = torch.acos(directions[:, 2].clamp(-0.999999, 0.999999))
    phi = torch.atan2(directions[:, 1], directions[:, 0])
    fractions = torch.full((observed.shape[0], config.nfibres), 0.05,
                           device=observed.device)
    fractions[:, 0] = 0.55
    return {"s0": s0, "d": d, "d_std": (d * 0.5).clamp_min(1e-5),
            "f": fractions, "th": theta, "ph": phi}


@torch.no_grad()
def _sample_chunk(observed, bvals, bvecs, config, generator):
    state = _initial_state(observed, bvals, bvecs, config)
    current_energy = _energy(state, observed, bvals, bvecs, config)
    nvoxels = observed.shape[0]
    nsamples = config.njumps // config.sample_every
    samples = {key: torch.empty((nvoxels, nsamples, config.nfibres),
                                device=observed.device)
               for key in ("th", "ph", "f")}
    d_samples = torch.empty((nvoxels, nsamples), device=observed.device)
    d_std_samples = torch.empty_like(d_samples) if config.model == 2 else None
    s0_samples = torch.empty_like(d_samples)
    scales = {key: (state[key] * 0.1)[:, None].clone()
              for key in ("s0", "d", "d_std")}
    scales.update({key: torch.full((nvoxels, config.nfibres), 0.2,
                                   device=observed.device)
                   for key in ("f", "th", "ph")})
    accepted = {key: torch.zeros_like(value) for key, value in scales.items()}

    def propose(key, column=None):
        nonlocal current_energy
        candidate = state.copy()
        old = state[key]
        if column is None:
            scale = scales[key][:, 0]
            candidate[key] = old + torch.randn(old.shape, generator=generator,
                                               device=old.device) * scale
            index = 0
        else:
            proposed = old.clone()
            proposed[:, column] += torch.randn((nvoxels,), generator=generator,
                                                device=old.device) * scales[key][:, column]
            candidate[key] = proposed
            index = column
        new_energy = _energy(candidate, observed, bvals, bvecs, config)
        draw = torch.rand((nvoxels,), generator=generator, device=old.device).clamp_min(1e-8)
        keep = torch.log(draw) < current_energy - new_energy
        state[key] = torch.where(keep[:, None], candidate[key], old) if column is not None \
            else torch.where(keep, candidate[key], old)
        current_energy = torch.where(keep, new_energy, current_energy)
        accepted[key][:, index] += keep

    for iteration in range(config.burnin + config.njumps):
        for key in ("d", "d_std", "s0"):
            if key != "d_std" or config.model == 2:
                propose(key)
        for fibre in range(config.nfibres):
            for key in ("th", "ph", "f"):
                propose(key, fibre)
        if (iteration + 1) % 40 == 0:
            for key, scale in scales.items():
                if key == "d_std" and config.model == 1:
                    continue
                scale.mul_(torch.sqrt((accepted[key] + 1) / (41 - accepted[key])))
                scale.clamp_(max=1e10)
                accepted[key].zero_()
        post = iteration + 1 - config.burnin
        if post > 0 and post % config.sample_every == 0:
            sample = post // config.sample_every - 1
            for key in samples:
                samples[key][:, sample] = state[key]
            d_samples[:, sample] = state["d"]
            s0_samples[:, sample] = state["s0"]
            if d_std_samples is not None:
                d_std_samples[:, sample] = state["d_std"]
    return samples, d_samples, d_std_samples, s0_samples


class TorchBEDPOSTX:
    """Fit one diffusion data set and write FSL-compatible posterior samples."""

    def __init__(self, device="cpu", threads=None, *, nfibres=3, model=2,
                 burnin=1000, njumps=1250, sample_every=25, ard_weight=1.0,
                 chunk_size=16384, seed=8665904):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if threads is not None:
            if not isinstance(threads, int) or threads < 1:
                raise ValueError("threads must be a positive integer")
            torch.set_num_threads(threads)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.config = BedpostXConfig(nfibres, model, burnin, njumps, sample_every,
                                    ard_weight, chunk_size, seed)

    @torch.no_grad()
    def __call__(self, subject_dir, output_dir=None, *, overwrite=False):
        started = time.perf_counter()
        subject_dir = Path(subject_dir).expanduser().resolve()
        output_dir = (Path(output_dir).expanduser().resolve() if output_dir is not None
                      else subject_dir.with_name(subject_dir.name + ".bedpostX"))
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        image = nib.load(str(subject_dir / "data.nii.gz"))
        mask_image = nib.load(str(subject_dir / "nodif_brain_mask.nii.gz"))
        if len(image.shape) != 4 or len(mask_image.shape) != 3:
            raise ValueError("data must be 4D and nodif_brain_mask must be 3D")
        if image.shape[:3] != mask_image.shape or not np.allclose(
                image.affine, mask_image.affine, atol=1e-4, rtol=0):
            raise ValueError("data and mask must share voxel shape and affine")
        bvals = np.loadtxt(subject_dir / "bvals", dtype=np.float32).reshape(-1)
        bvecs = np.loadtxt(subject_dir / "bvecs", dtype=np.float32)
        if bvecs.shape == (3, bvals.size):
            bvecs = bvecs.T
        if bvecs.shape != (bvals.size, 3) or bvals.size != image.shape[3]:
            raise ValueError("bvals/bvecs must match the number of DWI volumes")
        if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():
            raise ValueError("bvals and bvecs must be finite")
        if not np.any(bvals <= 50) or np.count_nonzero(bvals > 50) < 6:
            raise ValueError("data need at least one b0 and six diffusion directions")
        bvecs = bvecs / np.maximum(np.linalg.norm(bvecs, axis=1, keepdims=True), 1e-8)
        data = np.asarray(image.dataobj, dtype=np.float32)
        mask = np.asarray(mask_image.dataobj) > 0
        if not mask.any() or not np.isfinite(data[mask]).all():
            raise ValueError("mask must be nonempty and masked data finite")
        nvoxels = int(mask.sum())
        nsamples = self.config.njumps // self.config.sample_every
        bvals_t = torch.as_tensor(bvals.copy(), device=self.device)
        bvecs_t = torch.as_tensor(bvecs.copy(), device=self.device)
        sample_arrays = {key: np.zeros((self.config.nfibres, nvoxels, nsamples),
                                       dtype=np.float32) for key in ("th", "ph", "f")}
        d_array = np.zeros((nvoxels, nsamples), dtype=np.float32)
        s0_array = np.zeros_like(d_array)
        d_std_array = np.zeros_like(d_array) if self.config.model == 2 else None
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        voxels = data[mask]
        for start in range(0, nvoxels, self.config.chunk_size):
            stop = min(start + self.config.chunk_size, nvoxels)
            observed = torch.as_tensor(voxels[start:stop].copy(), device=self.device)
            samples, d, d_std, s0 = _sample_chunk(observed, bvals_t, bvecs_t,
                                                   self.config, generator)
            order = torch.argsort(samples["f"].mean(dim=1), dim=1, descending=True)
            for key, value in samples.items():
                value = torch.gather(value, 2, order[:, None, :].expand_as(value))
                sample_arrays[key][:, start:stop] = value.permute(2, 0, 1).cpu().numpy()
            d_array[start:stop] = d.cpu().numpy()
            s0_array[start:stop] = s0.cpu().numpy()
            if d_std_array is not None:
                d_std_array[start:stop] = d_std.cpu().numpy()

        output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            generated = ["run.json", "nodif_brain_mask.nii.gz",
                         "mean_dsamples.nii.gz", "mean_S0samples.nii.gz",
                         "mean_d_stdsamples.nii.gz"]
            for fibre in range(1, 4):
                generated.extend(f"merged_{key}{fibre}samples.nii.gz"
                                 for key in ("th", "ph", "f"))
                generated.extend((f"mean_f{fibre}samples.nii.gz",
                                  f"dyads{fibre}.nii.gz"))
            for name in generated:
                (output_dir / name).unlink(missing_ok=True)

        def save(name, values):
            frames = values.shape[-1]
            volume = np.zeros(mask.shape + (frames,), dtype=np.float32)
            volume[mask] = values
            if frames == 1 and name.startswith("mean_"):
                volume = volume[..., 0]
            header = image.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(volume, image.affine, header=header),
                     str(output_dir / f"{name}.nii.gz"))

        for fibre in range(self.config.nfibres):
            for key in ("th", "ph", "f"):
                save(f"merged_{key}{fibre + 1}samples", sample_arrays[key][fibre])
            save(f"mean_f{fibre + 1}samples", sample_arrays["f"][fibre].mean(axis=1,
                                                                               keepdims=True))
            theta = sample_arrays["th"][fibre]
            phi = sample_arrays["ph"][fibre]
            direction = np.stack((np.sin(theta) * np.cos(phi),
                                  np.sin(theta) * np.sin(phi), np.cos(theta)), axis=-1)
            dyad = np.einsum("vsi,vsj->vij", direction, direction) / nsamples
            _, vectors = np.linalg.eigh(dyad)
            save(f"dyads{fibre + 1}", vectors[:, :, -1])
        save("mean_dsamples", d_array.mean(axis=1, keepdims=True))
        save("mean_S0samples", s0_array.mean(axis=1, keepdims=True))
        if d_std_array is not None:
            save("mean_d_stdsamples", d_std_array.mean(axis=1, keepdims=True))
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), mask_image.affine,
                                 header=mask_image.header.copy()),
                 str(output_dir / "nodif_brain_mask.nii.gz"))
        elapsed = time.perf_counter() - started
        (output_dir / "run.json").write_text(json.dumps({
            "implementation": "fnit independent PyTorch BEDPOSTX model",
            "config": asdict(self.config), "device": str(self.device),
            "shape": list(image.shape), "nvoxels": nvoxels,
            "nsamples": nsamples, "elapsed_seconds": elapsed,
        }, indent=2) + "\n", encoding="utf-8")
        return BedpostXResult(output_dir, nvoxels, nsamples, elapsed)
