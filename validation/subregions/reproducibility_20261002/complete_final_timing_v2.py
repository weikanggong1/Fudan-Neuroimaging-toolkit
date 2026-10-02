"""CPU-only complete nested recipe timers after final all-run metrics succeed."""
from __future__ import annotations
import hashlib,json,sys,time,traceback
from pathlib import Path


def main():
    folder=Path(__file__).parent;analysis=folder/'final_all_analysis';output=analysis/'timing_complete'
    output.mkdir(parents=True,exist_ok=True)
    state_path=output/'v2_status.json'
    if state_path.exists():raise ValueError('Preserve an existing v2 timing status')
    extractor=folder/'extract_final_step_timing_v2.py'
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    state={'state':'waiting','scope':'CPU only, no fitting; corrected complete nested recipe timer mappings',
           'helper_sha256':sha(Path(__file__)),'actual_extractor_sha256':sha(extractor),'started_unix':time.time()}
    def save():
        temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
    save()
    try:
        while True:
            status=json.loads((analysis/'final_full_analysis_status.json').read_text())
            if status['state']=='failed':raise ValueError('Final full-pipeline analysis failed')
            if status['state']=='completed':break
            if time.time()-state['started_unix']>10800:raise TimeoutError('CPU timing deadline')
            time.sleep(10)
        from extract_final_step_timing_v2 import main as extract
        sys.argv=[str(extractor),'--final-result',str(analysis/'final_full_repeatability.json'),
                  '--official-timers',str(folder/'official_explicit_step_timers.json'),
                  '--official-metadata',str(folder/'queue.json'),
                  '--reconall-lineage',str(folder/'reconall_historical_lineage_audit.json'),'--output-dir',str(output)]
        extract()
        state.update(state='completed',finished_unix=time.time(),artifacts={name:{'bytes':(output/name).stat().st_size,'sha256':sha(output/name)} for name in ('final_step_timing.json','final_step_timing.tsv')})
        save();print(json.dumps(state),flush=True)
    except BaseException as error:
        state.update(state='failed',finished_unix=time.time(),error=f'{type(error).__name__}: {error}');save();traceback.print_exc();raise


if __name__=='__main__':main()
