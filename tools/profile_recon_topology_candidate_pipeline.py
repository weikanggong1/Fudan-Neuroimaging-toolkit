"""同输入候选链的透明阶段剖析；只在独立benchmark包装调用，不改算法。"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import pstats
from pathlib import Path
import benchmark_recon_topology_candidate_pipeline as benchmark


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['before-retess','initial-attributes','initial-global-priors','input-surface','input-qsphere','brain','after-statistics-reference','after-volume-reference','fitness-reference','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--threads',type=int,required=True)
    args=vars(parser.parse_args())
    # Only the diagnostic module's imported callable is wrapped. The production
    # function and all numeric/retess/ordering rules retain their original bytes.
    original=benchmark.evaluate_candidate
    records=[];profile=cProfile.Profile()
    def observed(**kwargs):
        result=original(**kwargs) if not records else profile.runcall(original,**kwargs)
        records.append(dict(candidate_seconds=result['seconds'],geometry_subseconds=result['geometry']['stage_seconds'],
                            cprofile_enabled=bool(len(records))))
        return result
    benchmark.evaluate_candidate=observed
    benchmark.run(**args)
    stats=pstats.Stats(profile)
    rows=[]
    for (filename,line,function),(primitive,total,self_seconds,cumulative,callers) in stats.stats.items():
        # Metrics only: no private source path, caller text, command or log.
        rows.append(dict(module=Path(filename).name,line=line,function=function,
                         primitive_calls=primitive,total_calls=total,self_seconds=self_seconds,cumulative_seconds=cumulative))
    rows.sort(key=lambda x:x['cumulative_seconds'],reverse=True)
    receipt=dict(scope='full exact frozen candidate repeated; second candidate transparently profiled, elapsed includes observer overhead',
                 observed_calls=records,top_cumulative_functions=rows[:50],all_function_count=len(rows),
                 profile_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 unprofiled_optimization_claim=False,complete_ga=False)
    (args['output']/'profile.metrics.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
