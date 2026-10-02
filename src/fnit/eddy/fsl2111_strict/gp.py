from __future__ import annotations

from dataclasses import dataclass
import math
import ctypes
import time
import numpy as np
import torch
import torch.nn.functional as F


def _fsl_amoeba(cost, start: np.ndarray, maxiter: int) -> np.ndarray:
    """FSL miscmaths Simplex/amoeba with its default unit starting steps."""
    points = np.vstack((start, start + np.eye(len(start))))
    values = np.array([cost(p) for p in points], dtype=np.float64)
    for _ in range(maxiter):
        best = int(np.argmin(values))
        worst = int(np.argmax(values))
        second_worst = max((i for i in range(len(values)) if i != worst), key=lambda i: values[i])
        if 2 * abs(values[worst] - values[best]) <= 1e-8 * (abs(values[worst]) + abs(values[best]) + 2e-16):
            break
        centroid = (points.sum(0) - points[worst]) / len(start)
        reflected = 2 * centroid - points[worst]
        reflected_value = cost(reflected)
        if reflected_value < values[worst]:
            points[worst], values[worst] = reflected, reflected_value
        if reflected_value <= values[best]:
            expanded = 2 * points[worst] - centroid
            expanded_value = cost(expanded)
            if expanded_value < values[worst]:
                points[worst], values[worst] = expanded, expanded_value
        elif reflected_value >= values[second_worst]:
            old_worst = values[worst]
            contracted = 0.5 * (points[worst] + centroid)
            contracted_value = cost(contracted)
            if contracted_value < values[worst]:
                points[worst], values[worst] = contracted, contracted_value
            if contracted_value >= old_worst:
                for i in range(len(points)):
                    if i != best:
                        points[i] = 0.5 * (points[i] + points[best])
                        values[i] = cost(points[i])
    return points[int(np.argmin(values))]


def _shell_groups(bvals: torch.Tensor, tolerance: float = 100.0):
    """Group DWI b-values into shells in ascending first-seen order."""
    vals = bvals.detach().cpu().double().numpy()
    groups: list[list[int]] = []
    means: list[float] = []
    assignments = np.empty(len(vals), dtype=np.int64)
    for idx, b in enumerate(vals):
        found = None
        for gi, m in enumerate(means):
            if abs(b - m) <= tolerance:
                found = gi
                break
        if found is None:
            found = len(groups); groups.append([]); means.append(float(b))
        groups[found].append(idx)
        means[found] = float(np.mean(vals[groups[found]]))
        assignments[idx] = found
    return groups, assignments, np.asarray(means, dtype=np.float64)


def _antipodal_angle_matrix(bvecs: torch.Tensor) -> torch.Tensor:
    g = bvecs / bvecs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    dot = (g @ g.T).abs().clamp(0, 1)
    return torch.acos(dot)


def _fsl_select_coordinates(mask: torch.Tensor, nvox: int, seed: int | None):
    """Replicate DataSelector's srand/rand coordinate draw on Linux/glibc.

    FSL 2111 calls srand(time(NULL)) when --initrand is 0. Pass seed=None to
    reproduce that behaviour, or an integer to make oracle comparisons exact.
    """
    libc = ctypes.CDLL(None)
    actual_seed = int(time.time()) if seed is None or int(seed) == 0 else int(seed)
    libc.srand(ctypes.c_uint(actual_seed))
    nx, ny, nz = map(int, mask.shape)
    # Keep glibc's draw and acceptance order; copy this fit's mask once rather
    # than synchronising a CUDA scalar for every rejected or accepted draw.
    mask_cpu = mask.detach().cpu().numpy()
    selected=[]; seen=set(); maxtry=int(1e8)
    for _ in range(maxtry):
        c=(libc.rand()%nx, libc.rand()%ny, libc.rand()%nz)
        if bool(mask_cpu[c]) and c not in seen:
            seen.add(c); selected.append(c)
            if len(selected)==nvox: break
    if len(selected)!=nvox:
        raise RuntimeError('unable to select requested unique GP voxels')
    return torch.as_tensor(selected, dtype=torch.long, device=mask.device), actual_seed

def _gaussian_kernel1d_fsl(sigma: float, n: int, device, dtype):
    x=torch.arange(n,device=device,dtype=dtype)-(n-1)/2
    k=torch.exp(-0.5*(x/sigma)**2)
    return k/k.sum()

def _selected_smoothed_data(data: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor, fwhm_mm: float, voxel_sizes):
    """Port DataSelector::get_smooth for selected coordinates."""
    if fwhm_mm <= 0:
        return data[:, coords[:,0], coords[:,1], coords[:,2]].T.contiguous()
    sig=[fwhm_mm/math.sqrt(8*math.log(2))/float(v) for v in voxel_sizes]
    ns=[int((x-0.001))*2+3 for x in sig]
    ks=[_gaussian_kernel1d_fsl(sig[d],ns[d],data.device,data.dtype) for d in range(3)]
    def smooth(x):
        for axis, kernel in enumerate(ks):
            shape=[1,1,1,1,1]; shape[axis+2]=len(kernel)
            pad=[0,0,0]; pad[axis]=len(kernel)//2
            x=F.conv3d(x,kernel.reshape(shape),padding=tuple(pad))
        return x

    mask_float=mask.to(data.dtype)[None,None]
    denominator=smooth(mask_float)[0,0]
    selected=[]
    for chunk in data.split(8):
        numerator=smooth((chunk*mask).unsqueeze(1))[:,0]
        selected.append((numerator[:,coords[:,0],coords[:,1],coords[:,2]] /
                         denominator[coords[:,0],coords[:,1],coords[:,2]].clamp_min(1e-20)).T)
    return torch.cat(selected,dim=1)

@dataclass
class NewSphericalGP:
    bvals: torch.Tensor
    bvecs: torch.Tensor
    shell_tolerance: float = 100.0
    ff: float = 10.0
    maxiter: int = 500

    def __post_init__(self):
        self.device = self.bvals.device
        self.dtype = torch.float64
        groups, assignments, means = _shell_groups(self.bvals, self.shell_tolerance)
        self.groups = groups
        self.group_index = torch.as_tensor(assignments, dtype=torch.long, device=self.device)
        self.group_b = torch.as_tensor(means, dtype=self.dtype, device=self.device)
        self.angles = _antipodal_angle_matrix(self.bvecs.to(self.dtype))
        log_b=torch.log(self.group_b)[self.group_index]
        self.log_b_difference=log_b[:,None]-log_b[None,:]
        self.hpar = None
        self.K = None
        self.iK = None
        self.means = None
        self.residual_data = None

    @property
    def n_groups(self):
        return len(self.groups)

    def _transform_hpar(self, hpar: torch.Tensor) -> torch.Tensor:
        return torch.exp(hpar)

    def _kernel(self, hpar: torch.Tensor, apply_ff: bool = False) -> torch.Tensor:
        th = self._transform_hpar(hpar)
        ratio=self.angles/th[1]
        K=torch.where(self.angles<th[1],th[0]*(1-1.5*ratio+0.5*ratio**3),0)
        if self.n_groups>1:
            K=K*torch.exp(-self.log_b_difference.square()/(2*th[2].square()))
        noise_offset=3 if self.n_groups>1 else 2
        noise=th[noise_offset+self.group_index]*(self.ff if apply_ff else 1.0)
        return K+torch.diag(noise)

    def _guess(self, selected_data: torch.Tensor) -> torch.Tensor:
        # selected_data: nvox x nscan after shell mean correction
        vars_ = []
        for inds in self.groups:
            x = selected_data[:, inds]
            v = x.var(dim=1, unbiased=True).mean().clamp_min(1e-12)
            vars_.append(float(v))
        mean_var=sum(vars_)/self.n_groups
        hp=[0.9*math.log(mean_var/3.0),0.45]
        if self.n_groups==1:
            hp.append(math.log(mean_var/3.0))
        else:
            hp.append(0.0)
            delta=0.2/(self.n_groups-1)
            hp.extend((1.1-i*delta)*math.log(mean_var/3.0)
                      for i in range(self.n_groups))
        result=torch.as_tensor(hp,device=self.device,dtype=self.dtype)
        torch.linalg.cholesky(self._kernel(result))
        return result

    def _cv_objective_np(self, p: np.ndarray, selected_data: torch.Tensor) -> float:
        hp = torch.as_tensor(p, dtype=self.dtype, device=self.device)
        K = self._kernel(hp, apply_ff=False)
        try:
            L = torch.linalg.cholesky(K)
            iK = torch.cholesky_inverse(L)
        except RuntimeError:
            return np.finfo(np.float64).max
        # each row is a selected voxel, y is scan vector
        q = selected_data @ iK.T
        ssd_vec = (q * q).sum(0)
        diag = torch.diagonal(iK)
        val = (ssd_vec / (diag * diag)).sum() / K.shape[0]
        ans = float(val.detach().cpu())
        return ans if math.isfinite(ans) else np.finfo(np.float64).max

    def fit(self, data: torch.Tensor, mask: torch.Tensor, nvox: int = 1000, seed: int | None = None, fwhm_mm: float = 0.0, voxel_sizes=None):
        """Fit FSL 2111 spherical GP to model-space DWI data.

        data: N x X x Y x Z. The caller supplies already-unwarped current scans.
        """
        if data.shape[0] != self.bvals.numel():
            raise ValueError("DWI scan count mismatch")
        data64 = data.to(self.dtype)
        means = torch.zeros((self.n_groups, *data.shape[1:]), dtype=self.dtype, device=data.device)
        residual = data64.clone()
        for gi, inds in enumerate(self.groups):
            means[gi] = data64[inds].mean(0)
            residual[inds] -= means[gi]
        self.means = means
        self.residual_data = residual
        available = int(mask.sum().item())
        if available == 0:
            raise ValueError("empty GP mask")
        if available < int(nvox):
            raise ValueError("FSL DataSelector requires nvoxhp <= valid voxel count")
        coords, actual_seed = _fsl_select_coordinates(mask, int(nvox), seed)
        self.data_selector_seed = actual_seed
        if voxel_sizes is None:
            voxel_sizes=(1.0,1.0,1.0)
        selected = _selected_smoothed_data(residual, mask, coords, fwhm_mm, voxel_sizes)
        hp0 = self._guess(selected)
        initial=hp0.detach().cpu().numpy()
        optimum = _fsl_amoeba(lambda p: self._cv_objective_np(p, selected), initial, int(self.maxiter))
        self.hpar_unfudged = torch.as_tensor(optimum, dtype=self.dtype, device=self.device)
        self.K_unfudged = self._kernel(self.hpar_unfudged, apply_ff=False)
        # FSL applies ff after hyperparameter optimisation by multiplying diagonal
        # error variances; hpar metadata changes, but prediction K is what matters.
        self.K = self._kernel(self.hpar_unfudged, apply_ff=True)
        L = torch.linalg.cholesky(self.K)
        self.iK = torch.cholesky_inverse(L)
        return self

    def prediction_weights(self, index: int, exclude: bool = False) -> torch.Tensor:
        if self.K is None:
            raise RuntimeError("GP not fitted")
        hp = self.hpar_unfudged
        th = self._transform_hpar(hp)
        angle=self.angles[index]
        ratio=angle/th[1]
        k=torch.where(angle<th[1],th[0]*(1-1.5*ratio+0.5*ratio**3),0)
        if self.n_groups>1:
            k=k*torch.exp(-self.log_b_difference[index].square()/(2*th[2].square()))
        if not exclude:
            return k @ self.iK
        keep = torch.ones(len(self.bvals), dtype=torch.bool, device=self.device); keep[index] = False
        Ksub = self.K[keep][:, keep]
        return k[keep] @ torch.linalg.inv(Ksub)

    def predict(self, index: int, exclude: bool = False) -> torch.Tensor:
        w = self.prediction_weights(index, exclude=exclude)
        gi = int(self.group_index[index])
        if exclude:
            keep = torch.ones(len(self.bvals), dtype=torch.bool, device=self.device); keep[index] = False
            r = self.residual_data[keep]
        else:
            r = self.residual_data
        pred = torch.einsum("n,nxyz->xyz", w.to(r.dtype), r)
        return (pred + self.means[gi]).to(torch.float32)

    def predict_all(self, exclude: bool = False) -> torch.Tensor:
        return torch.stack([self.predict(i, exclude=exclude) for i in range(len(self.bvals))])
