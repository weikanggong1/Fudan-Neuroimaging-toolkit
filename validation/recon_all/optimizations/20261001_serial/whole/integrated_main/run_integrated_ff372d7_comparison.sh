#!/bin/bash
set -euo pipefail
'/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python' '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/collect_hotspot_integrated_current.py' '--config' '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_integrated_ff372d7_comparison_config.json'
touch '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_integrated_ff372d7_comparison.finished'
