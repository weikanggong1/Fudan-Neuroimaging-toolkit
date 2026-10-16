# Fixed FreeSurfer 8.2 `mris_sphere -q` stage

The `-q` path is implemented in
`src/fnit/recon_all/sphere_python.py` and
`src/fnit/recon_all/sphere_quick_python.py`. The first file
recreates 300 inflation updates and their carried vertex momentum; the second
performs the four-epoch nonlinear-area optimizer. For the fixed bilateral
`fs_sub01/inflated.nofix` inputs, the independent NumPy CPU chain reproduces
all 102,764 LH and 101,454 RH final `qsphere.nofix` vertex coordinates and
205,560/202,936 ordered faces exactly. It takes 139.00/146.40 s on control-host
in one run per hemisphere. See
[`SPHERE_QUICK_STATUS.md`](SPHERE_QUICK_STATUS.md) for intermediate checks,
source provenance, timing limits, and the RH status.

This geometry port is not integrated into a native-free recon-all runner or CUDA implementation, and has not established multi-subject parity. Its actual bilateral CLI outputs also match the official quick spheres when fed the independently generated Python `inflated` files; see the [connected chain report](inflate_to_qsphere_chain_exact_report.json). A longer bilateral [six-stage replay](SMOOTH_SURFACE.md) begins at frozen
`filled.mgz` and `norm.mgz` and reaches the same exact quick spheres using
only Python stage outputs. The later conventional `mris_sphere` call that
creates `?h.sphere` for registration is a separate algorithm and remains
[open](SPHERE_STANDARD_STATUS.md).
