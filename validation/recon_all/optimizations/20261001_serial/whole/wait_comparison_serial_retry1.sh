set -eu
while ! test -f /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole.retry1.finished; do
 if ! tmux has-session -t fnit-serial-whole-retry1 2>/dev/null; then
  sleep 30
  if ssh -S /tmp/fnit-gpucw1-gongwk@gpucw1:22 -oBatchMode=yes gongwk@gpucw1 'test -f /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole.retry1.finished'; then break; fi
  echo "whole retry1 execution failed; comparison blocked"; exit 1
 fi
 sleep 15
done
ssh -S /tmp/fnit-gpucw1-gongwk@gpucw1:22 -oBatchMode=yes gongwk@gpucw1 '/cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/fnit_main_env/bin/python /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/collect_hotspot_whole_comparison.py --config /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/whole_comparison_retry1_config.json'
touch /cwStorage/home/gongwk/Notebook_code/freesurfer_synth/work/recon_standard_20260929/serial_20261001/comparison.retry1.finished
