from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FSL2111Config:
    """Pinned subset of FSL EDDY 2111.0 used by UK Biobank v1.5.

    This backend intentionally targets the volume-to-volume command used in the
    FNIT validation harness, not the complete modern eddy feature surface.
    """

    niter: int = 8
    fwhm_mm: tuple[float, ...] = (10.0, 8.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0)
    ff: float = 10.0
    nvoxhp: int = 1000
    ol_nstd: float = 4.0
    ol_nvox: int = 250
    ref_scan_no: int = 0
    gp_nm_maxiter: int = 500
    b0_threshold: float = 100.0
    shell_tolerance: float = 100.0
    use_tf32: bool = True
    spline_precision: float = 1e-8
    enable_post_eddy_shell_alignment: bool = True

    def __post_init__(self):
        if self.niter != len(self.fwhm_mm):
            raise ValueError("FSL2111Config requires one FWHM value per outer iteration")
        if self.niter <= 0:
            raise ValueError("niter must be positive")
        if self.nvoxhp <= 0:
            raise ValueError("nvoxhp must be positive")
        if self.ol_nvox <= 0:
            raise ValueError("ol_nvox must be positive")
        if self.ff < 1.0:
            raise ValueError("ff must be >= 1")


FSL_EDDY_VERSION = "2111.0"
FSL_EDDY_COMMIT = "ecfef26151c2613d0f4e1b45dcbbe100b58db50c"
FSL_REFERENCE_COMMAND = (
    "eddy_cuda10.2 --flm=quadratic --resamp=jac --slm=linear --niter=8 "
    "--fwhm=10,8,4,2,0,0,0,0 --ff=10 --sep_offs_move --nvoxhp=1000 --repol --rms"
)
