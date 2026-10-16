#!/bin/bash
set -euo pipefail
'/data/workspaces/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python' '/data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/candidate5b/validation/recon_all/python_gpu_port/collect_hotspot_whole_comparison.py' '--config' '/data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_affine_29f_comparison_config.json'
touch '/data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_affine_29f_comparison.finished'
