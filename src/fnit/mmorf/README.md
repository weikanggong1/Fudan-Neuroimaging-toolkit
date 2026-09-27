# MMORF module

The public single-subject function is `run_mmorf`; the CLI and dMRI pipeline
call this same function. `TorchMMORF` is the lower-level in-memory interface,
and `apply_mmorf_warp` applies its reference-grid voxel-displacement warp. See
docs/mmorf/README.md for the exact coordinate contract, inputs, outputs,
official MMORF command, algorithm differences, and benchmark.
