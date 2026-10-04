"""Isolated explicitly fused oneDNN op on one captured complete real SR layer.

The input is the previously captured first ELU, not a replacement whole-graph
oracle. Both the fused Conv+Bias and fused Conv+Bias+ELU are separate graphs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import re
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sr_first_layer import digest, metrics


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('reference', 'candidate'), required=True)
    for name in ('first-dir', 'weights', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--official-script', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('preserve previous evidence')
    args.output_dir.mkdir(parents=True)
    assert args.weights.stat().st_size == 106163752
    assert digest(args.weights) == 'a472f776e7b33b5ea6e10c801f55fee488f1477a208b3e6998dc1aec1d9c5f8b'
    source = np.load(args.first_dir/'elu.npy')
    report = {'arm': args.arm, 'scope': __doc__, 'input_file_sha256': digest(args.first_dir/'elu.npy'),
              'input_array_sha256': array_sha(source), 'driver_sha256': digest(__file__), 'stages': {}}
    if args.arm == 'reference':
        assert digest(args.official_script) == '917b20e90b00c1b2807ffcc1e8e2db123d3a63d52c8d395b8ba5a7f078c8b098'
        namespace = runpy.run_path(str(args.official_script), run_name='_isolated_reference_')
        tf = namespace['tf']
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from tensorflow.python.framework import op_def_library, op_def_registry
        from tensorflow.core.framework import attr_value_pb2
        definition = op_def_registry.get('_MklNativeFusedConv3D')
        if definition is None:
            raise RuntimeError('reference binary has no registered native fused Conv3D op')
        model = namespace['build_model'](str(args.weights))
        kernel, bias = model.get_layer('unet_conv_downarm_0_1').get_weights()
        input_value = tf.constant(np.ascontiguousarray(source.transpose(0,2,3,4,1)))
        kernel_value, bias_value = tf.constant(kernel), tf.constant(bias)
        report['tensorflow_version'] = tf.__version__
        report['registered_op_definition'] = str(definition)
        profile = args.output_dir/'profile.private'
        tf.profiler.experimental.start(str(profile), tf.profiler.experimental.ProfilerOptions(host_tracer_level=2, python_tracer_level=0, device_tracer_level=0))
        try:
            for name, fused_ops in (('raw', ['BiasAdd']), ('elu', ['BiasAdd', 'Elu'])):
                @tf.function(autograph=False)
                def fused(value):
                    _, _, operation, outputs = op_def_library._apply_op_helper(
                        '_MklNativeFusedConv3D', input=value, filter=kernel_value, args=[bias_value],
                        strides=[1]*5, padding='SAME', data_format='NDHWC', dilations=[1]*5,
                        fused_ops=fused_ops, is_filter_const=True, padding_list=[],
                        epsilon=.0001, leakyrelu_alpha=.2, name='isolated_fused_reference')
                    # This binary registers this internal op only with the
                    # MklNameChangeOp label, normally inserted by the rewrite.
                    operation._set_attr('_kernel', attr_value_pb2.AttrValue(s=b'MklNameChangeOp'))
                    return outputs[0]
                started = time.perf_counter()
                output = fused(input_value).numpy().transpose(0,4,1,2,3)
                report['stages'][name] = {'seconds': time.perf_counter()-started, 'array_sha256': array_sha(output),
                                         'fused_ops': fused_ops, 'shape': list(output.shape),
                                         'kernel_label': 'MklNameChangeOp'}
                np.save(args.output_dir/(name+'.npy'), output)
                del output
        finally:
            tf.profiler.experimental.stop()
        from tensorflow.core.profiler.protobuf import xplane_pb2
        names = []
        for path in profile.rglob('*.xplane.pb'):
            space = xplane_pb2.XSpace(); space.ParseFromString(path.read_bytes())
            for plane in space.planes:
                names.extend(item.name for item in plane.event_metadata.values() if 'Conv' in item.name or 'Elu' in item.name)
        report['actual_execution_event_names'] = sorted(set(names))
    else:
        import torch
        from numba import set_num_threads
        from fnit.synthsr.model import SynthSRUNet, load_h5_weights
        from elu_onednn_numba import elu_torch, _elu_flat
        torch.set_num_threads(8); set_num_threads(8)
        model = load_h5_weights(SynthSRUNet(), args.weights).eval().to(memory_format=torch.channels_last_3d)
        report['helper_sha256'] = digest(Path(__file__).with_name('elu_onednn_numba.py'))
        report['production_model_sha256'] = digest(sys.modules['fnit.synthsr.model'].__file__)
        reference = json.loads((args.reference_dir/'report.public.json').read_text())
        assert report['input_file_sha256'] == reference['input_file_sha256']
        with torch.inference_mode():
            value = torch.from_numpy(source).to(memory_format=torch.channels_last_3d)
            started = time.perf_counter(); raw = model.down[0][1](value)
            report['stages']['raw'] = {'seconds': time.perf_counter()-started,
                                     'metrics': metrics(np.load(args.reference_dir/'raw.npy', mmap_mode='r'), raw.numpy())}
            started = time.perf_counter(); output = elu_torch(raw)
            report['stages']['elu'] = {'seconds': time.perf_counter()-started,
                                     'metrics': metrics(np.load(args.reference_dir/'elu.npy', mmap_mode='r'), output.numpy()),
                                     'array_sha256': array_sha(output.numpy())}
            # Separates ELU arithmetic from convolution accumulation differences.
            reference_raw = torch.from_numpy(np.load(args.reference_dir/'raw.npy'))
            started = time.perf_counter(); reference_input_output = elu_torch(reference_raw)
            report['stages']['elu_from_reference_raw'] = {'seconds': time.perf_counter()-started,
                'metrics': metrics(np.load(args.reference_dir/'elu.npy', mmap_mode='r'), reference_input_output.numpy())}
        llvm='\n'.join(_elu_flat.inspect_llvm().values())
        assembly='\n'.join(_elu_flat.inspect_asm().values())
        report['jit_evidence']={'llvm_sha256':hashlib.sha256(llvm.encode()).hexdigest(),
                                'assembly_sha256':hashlib.sha256(assembly.encode()).hexdigest(),
                                'explicit_fma_intrinsic_present':'llvm.fma' in llvm,
                                'assembly_fma_count':len(re.findall(r'\bv(?:f[mn]add|fmsub|fnmsub)\w*\b',assembly)),
                                'fastmath':False,
                                'cpu_flags':Path('/proc/cpuinfo').read_text().split('flags',1)[1].split('\n',1)[0].split()}
        (args.output_dir/'elu_llvm.private.txt').write_text(llvm)
        (args.output_dir/'elu_assembly.private.txt').write_text(assembly)
    report['status'] = 'complete'
    (args.output_dir/'report.public.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
