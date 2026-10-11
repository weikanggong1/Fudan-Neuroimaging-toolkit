"""完整自产PDF→成熟HISTO mean，保存后事后比较实际候选源参数。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from fnit.recon_all.topology_curvature_histogram import _source_mean
from benchmark_recon_topology_match_fields import load_match_capture


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(*, candidate, prior_report, reference, reference_repeat, output):
    """输入为已自产PDF及规范MAT1只读捕获；输出2个F32均值及严格位诊断。"""
    output=Path(output)
    if output.exists():raise FileExistsError('new output directory required')
    previous=json.loads(Path(prior_report).read_text())
    if previous['status']!='complete_global_priors':raise ValueError('complete same-domain priors required')
    pdf=np.load(candidate,allow_pickle=False)
    values=np.asarray([_source_mean(pdf[name+'_bins'],pdf[name+'_counts'],pdf[name+'_bin_size'][0])
                       for name in ('k1','k2')],np.float32)
    output.mkdir(parents=True)
    np.savez(output/'candidate_means.npz',means=values)
    # Candidate values are frozen before either reference is opened.
    before=load_match_capture(capture_path=reference)
    again=load_match_capture(capture_path=reference_repeat)
    source=before['parameters'][2:4];repeat=again['parameters'][2:4]
    exact=values.tobytes()==source.tobytes()
    report=dict(schema_version=1,scope='self-generated global PDFs -> source HISTO means, benchmark only',
        status='complete_prior_means' if exact else 'failed_strict_prior_means',
        candidate_values=values.tolist(),reference_values=source.tolist(),
        candidate_f32_bits=values.view(np.uint32).tolist(),reference_f32_bits=source.view(np.uint32).tolist(),
        differing_values=int(np.count_nonzero(values!=source)),bitwise_equal=exact,
        maximum_absolute_error=float(np.max(np.abs(values.astype(np.float64)-source.astype(np.float64)))),
        reference_repeat_bitwise_equal=source.tobytes()==repeat.tobytes(),
        candidate_checkpoint_sha256=sha(candidate),prior_report_sha256=sha(prior_report),
        reference_capture_sha256=sha(reference),reference_repeat_capture_sha256=sha(reference_repeat),
        script_sha256=sha(__file__),mean_module_sha256=sha(__import__(
            'fnit.recon_all.topology_curvature_histogram',fromlist=['']).__file__),
        parser_module_sha256=sha(__import__('benchmark_recon_topology_match_fields').__file__),
        own_mean_checkpoint_sha256=sha(output/'candidate_means.npz'),
        reference_after_only=True,reference_values_consumed_for_mean=False,complete_ga=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],bitwise_equal=exact)))
    return 0 if exact else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','prior-report','reference','reference-repeat','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(verify(**vars(parser.parse_args())))
