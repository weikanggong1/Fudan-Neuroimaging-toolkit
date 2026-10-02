#!/bin/bash
set -euo pipefail
'/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python' '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/candidate5b/validation/recon_all/python_gpu_port/collect_hotspot_whole_comparison.py' '--config' '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_affine_29f_comparison_config.json'
touch '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_affine_29f_comparison.finished'
