set -eu
while ! test -f /data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole.retry1.finished; do
 if ! tmux has-session -t fnit-serial-whole-retry1 2>/dev/null; then
  sleep 30
  if ssh -S /tmp/fnit-accelerator-a-1-user@accelerator-a-1:22 -oBatchMode=yes user@accelerator-a-1 'test -f /data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole.retry1.finished'; then break; fi
  echo "whole retry1 execution failed; comparison blocked"; exit 1
 fi
 sleep 15
done
ssh -S /tmp/fnit-accelerator-a-1-user@accelerator-a-1:22 -oBatchMode=yes user@accelerator-a-1 '/data/workspaces/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python /data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/collect_hotspot_whole_comparison.py --config /data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_comparison_retry1_config.json'
touch /data/workspaces/freesurfer_synth/work/recon_standard_20260929/serial_20261001/comparison.retry1.finished
