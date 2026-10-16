#!/bin/bash
set -euo pipefail
'/data/workspaces/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python' '/data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/collect_hotspot_integrated_current.py' '--config' '/data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_integrated_ff372d7_comparison_config.json'
touch '/data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_integrated_ff372d7_comparison.finished'
