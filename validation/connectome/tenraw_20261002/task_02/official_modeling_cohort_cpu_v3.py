"""Independent official corrected-DWI -> masks -> response -> CSD -> normalise.

Reference only. No FNIT imports, torch, GPU computation or frozen solver inputs.
The upstream readiness contract must bind a new independent official rawprep.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
from nibabel.orientations import (apply_orientation, axcodes2ornt, io_orientation,
                                 inv_ornt_aff, ornt_transform)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


EDDY_SCIENTIFIC_FLAGS = ['--flm=quadratic','--resamp=jac','--slm=linear','--niter=8',
                         '--fwhm=10,8,4,2,0,0,0,0','--ff=10','--sep_offs_move',
                         '--nvoxhp=1000','--repol','--rms','--initrand=12345']
EDDY_BINARY_SHA = {'cpu':'a5cf07367c42a6215ede24705ee8e10c6e76df8b170ac67fb8cb25f0c3d30f1f',
                   'gpu':'c6dcf94c9c8826c74039885b579ef7117996a52ae3efb01ed91308d4b910d591'}


def validate_eddy_solver(report, root):
    solver = report.get('EDDY_solver', 'gpu')
    if solver not in ('cpu','gpu'):
        raise ValueError('Unrecognised actual EDDY solver')
    label = 'official_EDDY_CPU' if solver == 'cpu' else 'official_EDDY_GPU'
    stages = {record['stage']:record for record in report['commands']}
    for stage in ('official_topup','official_synthstrip_CPU',label):
        if stage not in stages or stages[stage].get('returncode') != 0:
            raise ValueError(f'Upstream {stage} incomplete')
    stage = stages[label]; command = stage['command']; binary = Path(command[0])
    expected_name = 'eddy_cpu' if solver == 'cpu' else 'eddy_cuda10.2'
    if (binary.name != expected_name or stage['program_sha256'] != EDDY_BINARY_SHA[solver]
            or sha(binary) != stage['program_sha256']):
        raise ValueError('EDDY actual binary name/SHA does not match solver')
    if report.get('gp_seed') != 12345 or report.get('threads') != 8:
        raise ValueError('EDDY fixed GP seed/CPU threads changed')
    for flag in EDDY_SCIENTIFIC_FLAGS + [f"--ref_scan_no={report['selection']['ap_index']}"]:
        key = flag.split('=')[0]
        observed = [value for value in command[1:] if value.split('=')[0] == key]
        if observed != [flag]:
            raise ValueError(f'EDDY scientific flag differs or duplicates: {key}')
    if solver == 'cpu':
        if ('GPU_UUID' not in report or report['GPU_UUID'] is not None or stage.get('GPU') is not False or
                stage.get('env',{}).get('CUDA_VISIBLE_DEVICES') != ''):
            raise ValueError('CPU solver carries GPU labels/environment')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
            if stage['env'].get(key) != '8':
                raise ValueError('CPU solver thread environment differs')
    elif (report.get('GPU_UUID') != 'GPU-26e41f63-1a65-6b3e-5370-fa9a2934ca8e'
          or stage.get('GPU') is not True):
        raise ValueError('GPU solver label/UUID differs')
    paths = {'imain':root/'raw/AP.nii.gz','mask':root/'mask/nodif_brain_mask.nii.gz',
             'acqp':root/'topup/acqparams.txt','index':root/'eddy/eddy_index.txt',
             'bvecs':root/'raw/AP.bvec','bvals':root/'raw/AP.bval',
             'topup':root/'topup/fieldmap_out','out':root/'eddy/data'}
    for key,path in paths.items():
        observed = [value for value in command[1:] if value.startswith('--'+key+'=')]
        if observed != [f'--{key}={path}']:
            raise ValueError('EDDY did not consume complete own official inputs')
    return {'solver':solver,'stage':label,'GPU_UUID':report.get('GPU_UUID'),
            'binary':str(binary),'binary_sha256':stage['program_sha256'],
            'actual_command':command,'actual_host':stage['host'],
            'wall_seconds':stage['wall_seconds'],'CPU_threads':8}


def validate_cpu_completed_sidecar(path, report, root):
    sidecar = root/'completed_contract.json'
    if not sidecar.is_file():
        raise ValueError('CPU completed_contract missing')
    d = json.loads(sidecar.read_text())
    if (d.get('completed') is not True or d.get('EDDY_solver') != 'cpu' or
            'GPU_UUID' not in d or d['GPU_UUID'] is not None or d.get('subject') != report['subject'] or
            d.get('raw_sha256') != report['input_sha256'] or
            d.get('output_sha256') != report['output_sha256'] or d.get('GP_seed') != 12345):
        raise ValueError('CPU completed_contract labels/input/output SHA disagree')
    for key,relative in {'data':'eddy/data.nii.gz','rotated_bvecs':'eddy/data.eddy_rotated_bvecs',
                         'mask':'mask/nodif_brain_mask.nii.gz','bvals':'raw/AP.bval',
                         'topup_prefix':'topup/fieldmap_out'}.items():
        if d.get(key) != str(root/relative):
            raise ValueError('CPU completed_contract points outside own chain')
    verified = root/'completed_contract_verified.json'
    result = {'path':str(sidecar),'sha256':sha(sidecar)}
    if report.get('source_CPU_stage_lineage'):
        if not verified.is_file():
            raise ValueError('CPU verified completed sidecar missing')
        v = json.loads(verified.read_text())
        for key in ('completed','subject','EDDY_solver','GPU_UUID','GP_seed','raw_sha256','output_sha256','data','rotated_bvecs','bvals','mask','topup_prefix','ref_scan_no'):
            if v.get(key) != d.get(key):
                raise ValueError('CPU verified completed sidecar differs')
        if v.get('report') != str(path) or v.get('report_sha256') != sha(path):
            raise ValueError('CPU verified report SHA differs')
        result.update(verified_path=str(verified),verified_sha256=sha(verified))
    return result


def validate_restored_CPU_lineage(report, root):
    lineage = report.get('source_CPU_stage_lineage')
    if not lineage:
        return None
    source = Path(lineage['source'])
    source_report = Path(lineage['report'])
    if (lineage.get('CPU_stage_reexecuted') is not False or
            lineage.get('restored_successful_own_official_TOPUP_and_SynthStrip') is not True or
            source_report.parent != source or sha(source_report) != lineage['report_sha256']):
        raise ValueError('Restored CPU stage source report/labels changed')
    original = json.loads(source_report.read_text())
    original_stages = {stage['stage']:stage for stage in original['commands']}
    for stage in report['commands']:
        if stage['stage'] != 'official_EDDY_CPU' and original_stages.get(stage['stage']) != stage:
            raise ValueError('Restored CPU command lineage changed')
    for path,digest in lineage['source_EDDY_inputs_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Restored original own CPU input changed')
    for relative,digest in report['copied_successful_CPU_output_sha256'].items():
        if sha(root/relative) != digest or sha(source/relative) != digest:
            raise ValueError('Restored own CPU output bytes differ')
    pair = nib.load(root/'topup/B0_AP_PA.nii.gz')
    actual = nib.load(lineage['actual_FNIT_packing'])
    if (sha(lineage['actual_FNIT_packing']) != lineage['actual_FNIT_packing_sha256'] or
            not np.array_equal(pair.affine,actual.affine) or
            not np.array_equal(np.asarray(pair.dataobj),np.asarray(actual.dataobj))):
        raise ValueError('Restored raw pair differs from actual formal packing')
    for stem,index,frame in [('AP',report['selection']['ap_index'],0),('PA',report['selection']['pa_index'],1)]:
        raw = nib.load(root/'raw'/f'{stem}.nii.gz')
        bvals = np.loadtxt(root/'raw'/f'{stem}.bval').reshape(-1)
        if bvals[index] >= 100 or not np.array_equal(np.asarray(raw.dataobj)[...,index],np.asarray(pair.dataobj)[...,frame]):
            raise ValueError('Restored pair differs from actual canonical raw frame')
    return lineage


def ready_contract(path, forbidden_roots):
    if not Path(path).is_file():
        return None
    d = json.loads(Path(path).read_text())
    if d.get('completed') is not True:
        if d.get('state') == 'failed':
            raise ValueError('Bound upstream official rawprep failed; use a new explicit namespace')
        return None
    if d.get('kind') != 'independent_official_rawprep':
        budget_scope = d.get('scope','').startswith('official CPU reference: restored own successful raw TOPUP/SynthStrip lineage')
        if budget_scope and not (Path(path).parent/'completed_contract_verified.json').is_file():
            return None
        if not d.get('scope', '').startswith('independent official rawprep') and not budget_scope:
            raise ValueError('Readiness must identify the new independent official rawprep')
        root = Path(path).parent.resolve()
        solver_metadata = validate_eddy_solver(d, root)
        restored_lineage = validate_restored_CPU_lineage(d,root) if budget_scope else None
        completed_sidecar = validate_cpu_completed_sidecar(path,d,root) if solver_metadata['solver']=='cpu' else None
        relative = {'dwi':'eddy/data.nii.gz', 'bvecs':'eddy/data.eddy_rotated_bvecs',
                    'topup_fieldcoef':'topup/fieldmap_out_fieldcoef.nii.gz',
                    'topup_movpar':'topup/fieldmap_out_movpar.txt',
                    'eddy_mask':'mask/nodif_brain_mask.nii.gz'}
        files = {key:str(root/value) for key,value in relative.items()}
        hashes = {key:d['output_sha256'][value] for key,value in relative.items()}
        for key in ('topup_fieldcoef','topup_movpar','eddy_mask'):
            if d['EDDY_inputs_sha256'][files[key]] != hashes[key]:
                raise ValueError('Official stage dependency digest differs')
        files.update(bvals=str(root/'raw/AP.bval'), upstream_report=str(Path(path).resolve()))
        hashes['bvals'] = d['input_sha256'][str((root/'raw/AP.bval').resolve())]
        hashes['upstream_report'] = sha(path)
        d = {'kind':'independent_official_rawprep', 'completed':True, 'subject':d['subject'],
             'files':files, 'sha256':hashes, 'actual_rawprep_report':str(path),
             'actual_EDDY':solver_metadata, 'completed_rawprep_sidecar':completed_sidecar,
             'raw_frame_selection':d['selection'],'restored_CPU_stage_lineage':restored_lineage}
    required = ('dwi', 'bvecs', 'bvals', 'topup_fieldcoef', 'topup_movpar', 'eddy_mask', 'upstream_report')
    for key in required:
        p = Path(d['files'][key]).resolve()
        if any(p.is_relative_to(Path(root).resolve()) for root in forbidden_roots):
            raise ValueError(f'{key} points into a forbidden old/FNIT namespace')
        if sha(p) != d['sha256'][key]:
            raise ValueError(f'{key}: readiness SHA differs')
    return d


def validate_raw_binding(contract, config):
    """Require exactly the ten canonical raw files, then independently hash them."""
    report = json.loads(Path(contract['files']['upstream_report']).read_text())
    manifest = json.loads(Path(config['manifest']).read_text())
    cases = [case for case in manifest['cases'] if case['subject'].removeprefix('sub-') == report['subject'].removeprefix('sub-')]
    if len(cases) != 1:
        raise ValueError('Canonical subject is absent/duplicated')
    expected = {str(Path(config['raw_root']) / relative): value for relative,value in cases[0]['input_sha256'].items()}
    if len(expected) != 10 or report['input_sha256'] != expected:
        raise ValueError('Upstream absolute input_sha256 is not the exact canonical ten-file binding')
    if any(sha(path) != value for path,value in expected.items()):
        raise ValueError('Canonical raw acquisition changed')
    contract['canonical_raw_input_sha256'] = expected
    return contract


def model_case(config, subject, contract, contract_path):
    out = Path(config['output_root']) / ('sub-' + subject)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'report.json'
    report = {'kind': 'independent_official_modeling_chain', 'subject': subject,
              'profile': config['profile'], 'hostname': socket.gethostname(),
              'threads': 8, 'started_utc': utc(), 'state': 'running', 'execution_completed':False,
              'harness_sha256': sha(__file__), 'configuration': config,
              'upstream_contract': str(contract_path), 'upstream_contract_sha256': sha(contract_path),
              'upstream_files': contract['files'], 'upstream_sha256': contract['sha256'],
              'canonical_raw_input_sha256':contract['canonical_raw_input_sha256'],
              'actual_upstream_EDDY':contract['actual_EDDY'],
              'completed_rawprep_sidecar':contract.get('completed_rawprep_sidecar'),
              'raw_frame_selection':contract.get('raw_frame_selection'),
              'restored_CPU_stage_lineage':contract.get('restored_CPU_stage_lineage'),
              'alignment_binding':contract.get('alignment_binding'),
              'commands': [], 'cpu_operations': [], 'loadavg_start': os.getloadavg(),
              'environment': {'Python':sys.version, 'Python_binary_sha256':sha(sys.executable),
                              'nibabel':nib.__version__, 'numpy':np.__version__,
                              'cpu_affinity':sorted(os.sched_getaffinity(0)), 'CUDA_VISIBLE_DEVICES':''},
              'scope': 'independent own-official stage lineage; not matched-input solver isolation; rawprep wall remains upstream'}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (report['harness_sha256'] != sha(__file__) or report['configuration'] != config
                or report['upstream_contract_sha256'] != sha(contract_path)):
            raise ValueError('Existing reference report source/config/upstream changed')
        for operation in report['cpu_operations']:
            if sha(operation['output_path']) != operation['output_sha256']:
                raise ValueError('Recorded CPU orientation output changed')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               MKL_NUM_THREADS='8', FSLDIR=config['fsl_root'], FSLOUTPUTTYPE='NIFTI_GZ')
    env['PATH'] = config['mrtrix_bin'] + ':' + config['fsl_root'] + '/bin:' + str(Path(sys.executable).parent) + ':' + env.get('PATH', '')
    mr = Path(config['mrtrix_bin'])

    def run(label, binary, args, inputs, outputs, capture=None, mrtrix=True):
        inputs = {str(p): sha(p) for p in inputs}
        previous = next((r for r in report['commands'] if r['label'] == label), None)
        if previous and previous['returncode'] == 0:
            if previous['input_sha256'] != inputs or any(sha(p) != h for p, h in previous['output_sha256'].items()):
                raise ValueError(f'{label}: recorded input/output changed')
            return
        if any(Path(p).exists() for p in outputs):
            raise ValueError(f'{label}: partial/untracked output exists; preserve it and use a new namespace')
        binary = Path(binary)
        command = [str(binary), *map(str, args)] + (['-nthreads', '8'] if mrtrix else [])
        version = subprocess.run([str(binary), '-version'], env=env, capture_output=True, text=True) if mrtrix else None
        record = {'label': label, 'command': command, 'input_sha256': inputs,
                  'binary_sha256': sha(binary), 'resolved_binary_sha256': sha(binary.resolve()),
                  'version': version.stdout + version.stderr if version else (Path(config['fsl_root']) / 'etc/fslversion').read_text(),
                  'host': socket.gethostname(), 'started_utc': utc()}
        start = time.perf_counter()
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(command, env=env, stdout=subprocess.PIPE if capture else log,
                                    stderr=log if capture else subprocess.STDOUT, text=True)
        record.update(wall_s=time.perf_counter() - start, returncode=result.returncode)
        if capture and result.returncode == 0:
            Path(capture).write_text(result.stdout)
        record['output_sha256'] = {str(p): sha(p) for p in outputs if Path(p).is_file()}
        report['commands'].append(record)
        save(report_path, report)
        if result.returncode or len(record['output_sha256']) != len(outputs):
            report['state'] = 'failed'; save(report_path, report)
            raise RuntimeError(f'{label} failed; see its isolated log')

    files = contract['files']; dwi = Path(files['dwi']); bvecs = Path(files['bvecs']); bvals = Path(files['bvals'])
    grad = out / 'gradient_mrtrix.txt'; shells_file = out / 'shells.txt'
    run('gradient', mr / 'mrinfo', [dwi, '-fslgrad', bvecs, bvals, '-bvalue_scaling', 'no', '-export_grad_mrtrix', grad], [dwi, bvecs, bvals], [grad])
    run('shells', mr / 'mrinfo', [dwi, '-grad', grad, '-shell_bvalues'], [dwi, grad], [shells_file], capture=shells_file)
    shells = np.fromstring(shells_file.read_text(), sep=' ')
    if len(shells) < 2 or not np.isfinite(shells).all():
        raise ValueError('Invalid official shell centres')
    report['observed_shell_bvalues'] = shells.tolist()
    # b<50 is FNIT mean_bzero's fixed definition, independent from raw TOPUP b<100 frame selection.
    indices = np.flatnonzero(np.loadtxt(bvals).reshape(-1) < 50)
    if not len(indices):
        raise ValueError('No b<50 volume')
    bzero = out / 'bzero.nii.gz'; mean = out / 'mean_bzero.nii.gz'
    run('bzero', mr / 'mrconvert', [dwi, bzero, '-coord', '3', ','.join(map(str, indices)), '-datatype', 'float32'], [dwi, bvals], [bzero])
    run('mean_bzero', mr / 'mrmath', [bzero, 'mean', mean, '-axis', '3'], [bzero], [mean])
    mean_las = out / 'mean_bzero_LAS.nii.gz'; brain = out / 'brain_mask.nii.gz'
    mean_brain = out / 'mean_bzero_brain.nii.gz'
    if not mean_las.exists():
        start = time.perf_counter(); image = nib.load(mean)
        orientation = ornt_transform(io_orientation(image.affine), axcodes2ornt(('L', 'A', 'S')))
        values = np.ascontiguousarray(apply_orientation(image.get_fdata(dtype=np.float32), orientation))
        affine = image.affine @ inv_ornt_aff(orientation, image.shape)
        nib.save(nib.Nifti1Image(values, affine), mean_las)
        report['cpu_operations'].append({'label': 'LAS_view_for_official_BET', 'wall_s': time.perf_counter()-start,
                                         'input_sha256': sha(mean), 'output_path':str(mean_las), 'output_sha256': sha(mean_las)})
        save(report_path, report)
    bet_prefix = out / 'bet_LAS'; bet_mask = out / 'bet_LAS_mask.nii.gz'
    run('BET', Path(config['fsl_root']) / 'bin/bet', [mean_las, bet_prefix, '-m', '-f', '0.2', '-g', '-0.05', '-R'], [mean_las], [bet_mask, out / 'bet_LAS.nii.gz'], mrtrix=False)
    if not brain.exists():
        start = time.perf_counter(); image = nib.load(mean); mask = nib.load(bet_mask)
        orientation = ornt_transform(io_orientation(mask.affine), io_orientation(image.affine))
        values = np.ascontiguousarray(apply_orientation(np.asarray(mask.dataobj) > 0, orientation)).astype(np.uint8)
        aligned_affine = mask.affine @ inv_ornt_aff(orientation, mask.shape)
        if values.shape != image.shape or not np.allclose(aligned_affine, image.affine, rtol=0, atol=1e-5):
            raise ValueError('Official BET return-to-DWI geometry differs')
        nib.save(nib.Nifti1Image(values, image.affine), brain)
        bet_image = nib.load(out / 'bet_LAS.nii.gz')
        brain_values = np.ascontiguousarray(apply_orientation(bet_image.get_fdata(dtype=np.float32), orientation))
        nib.save(nib.Nifti1Image(brain_values, image.affine), mean_brain)
        report['cpu_operations'].append({'label': 'BET_mask_return_to_native', 'wall_s': time.perf_counter()-start,
                                         'input_sha256': sha(bet_mask), 'output_path':str(brain), 'output_sha256': sha(brain), 'brain_image_sha256':sha(mean_brain)})
        save(report_path, report)
    response_mask = out / 'response_mask.nii.gz'; fod_mask = out / 'fod_mask.nii.gz'; norm_mask = out / 'normalise_mask.nii.gz'
    run('dwi2mask', mr / 'dwi2mask', [dwi, response_mask, '-grad', grad], [dwi, grad], [response_mask])
    run('fod_mask', mr / 'maskfilter', [brain, 'dilate', fod_mask, '-npass', '2'], [brain], [fod_mask])
    run('normalise_mask', mr / 'maskfilter', [brain, 'erode', norm_mask, '-npass', '2'], [brain], [norm_mask])
    tensor = out / 'tensor.nii.gz'; fa = out / 'fa.nii.gz'; direction = out / 'direction.nii.gz'
    run('DTI', mr / 'dwi2tensor', [dwi, tensor, '-grad', grad, '-mask', brain, '-iter', '2'], [dwi, grad, brain], [tensor])
    run('FA', mr / 'tensor2metric', [tensor, '-fa', fa, '-vector', direction, '-modulate', 'none', '-mask', brain], [tensor, brain], [fa, direction])
    wmrf, gmrf, csfrf = [out / (name + 'rf.txt') for name in ('wm', 'gm', 'csf')]
    run('Dhollander', mr / 'dwi2response', ['dhollander', dwi, wmrf, gmrf, csfrf, '-grad', grad, '-mask', response_mask, '-scratch', out / 'response_scratch', '-nocleanup'], [dwi, grad, response_mask], [wmrf, gmrf, csfrf])
    wm, gm, csf = [out / (name + '.nii.gz') for name in ('wm', 'gm', 'csf')]
    run('MSMT_CSD', mr / 'dwi2fod', ['msmt_csd', dwi, wmrf, wm, gmrf, gm, csfrf, csf, '-grad', grad, '-mask', fod_mask, '-lmax', '8,0,0'], [dwi, grad, wmrf, gmrf, csfrf, fod_mask], [wm, gm, csf])
    norm = [out / (name + '_norm.nii.gz') for name in ('wm', 'gm', 'csf')]
    field = out / 'field.nii.gz'; accepted = out / 'accepted_mask.nii.gz'; factors = out / 'balance_factors.txt'
    run('mtnormalise', mr / 'mtnormalise', [wm, norm[0], gm, norm[1], csf, norm[2], '-mask', norm_mask, '-order', '3', '-niter', '15,7', '-reference', '0.28209479177', '-check_norm', field, '-check_mask', accepted, '-check_factors', factors], [wm, gm, csf, norm_mask], [*norm, field, accepted, factors])
    products = {'official_corrected_dwi':dwi, 'official_rotated_bvecs':bvecs,
                'official_bvals':bvals, 'official_gradient_mrtrix':grad,
                'official_shells':shells_file, 'official_mean_b0':mean,
                'official_mean_b0_brain':mean_brain, 'official_mean_b0_brain_mask':brain,
                'brain_mask':brain, 'response_mask':response_mask, 'fod_mask':fod_mask,
                'normalise_mask':norm_mask, 'FA':fa, 'principal_direction':direction,
                'wm_response':wmrf, 'gm_response':gmrf, 'csf_response':csfrf,
                'wm_fod_raw':wm, 'gm_raw':gm, 'csf_raw':csf,
                'wm_fod_normalized':norm[0], 'gm_normalized':norm[1], 'csf_normalized':norm[2],
                'normalise_field':field, 'accepted_mask':accepted, 'balance_factors':factors}
    geometry = nib.load(dwi); outputs = {}; readback_started = time.perf_counter()
    for key, path in products.items():
        item = {'path':str(path), 'size_bytes':path.stat().st_size, 'sha256':sha(path)}
        if str(path).endswith('.nii.gz'):
            image = nib.load(path)
            if image.shape[:3] != geometry.shape[:3] or not np.allclose(image.affine, geometry.affine, rtol=0, atol=1e-5):
                raise ValueError(f'{path.name}: output geometry differs')
            values = np.asanyarray(image.dataobj); finite = np.isfinite(values)
            item['grid'] = {'shape':list(image.shape), 'affine':image.affine.tolist(),
                            'spacing':list(map(float,image.header.get_zooms())),
                            'storage_dtype':str(image.get_data_dtype()), 'axcodes':list(nib.aff2axcodes(image.affine))}
            selected = values[finite]
            item['readback'] = {'decoded':True, 'values':int(values.size),
                                'finite':int(finite.sum()), 'nonfinite':int(values.size-finite.sum()),
                                'finite_min':float(selected.min()) if selected.size else None,
                                'finite_max':float(selected.max()) if selected.size else None}
            if key.endswith('mask') or key == 'official_mean_b0_brain_mask':
                if not np.isin(values,(0,1)).all() or not np.any(values):
                    raise ValueError(f'{key}: mask empty or nonbinary')
                item['readback']['nonzero_voxels'] = int(np.count_nonzero(values))
            del values,finite,selected
        else:
            values = np.loadtxt(path)
            item['readback'] = {'decoded':True, 'table_shape':list(values.shape),
                                'nonfinite':int(np.count_nonzero(~np.isfinite(values)))}
        outputs[key] = item
    for alias, key in {'corrected_dwi':'official_corrected_dwi', 'mean_b0':'official_mean_b0',
                       'mean_b0_brain':'official_mean_b0_brain'}.items():
        outputs[alias] = outputs[key]
    contract_path_out = out / 'consumer_contract.json'
    report['consumer_outputs'] = outputs
    report['consumer_contract_path'] = str(contract_path_out)
    report['readback_and_hash_wall_s'] = time.perf_counter()-readback_started
    report['actualcommands'] = report['commands']
    report.update(state='completed', execution_completed=True, completed_utc=utc(), loadavg_end=os.getloadavg())
    report['commands_wall_sum_s'] = sum(r['wall_s'] for r in report['commands'])
    save(report_path, report)
    consumer = {'schema_version':1, 'status':'completed', 'state':'completed',
                'case_id':'sub-'+subject, 'scope':'official_self_produced_raw_dwi_chain',
                'execution_completed':True, 'actualcommands':report['commands'],
                'upstream_report':{'path':str(contract_path), 'sha256':sha(contract_path)},
                'kind':'independent_official_modeling_chain',
                'output_id':Path(config['output_root']).name+'/sub-'+subject, 'subject':subject,
                'actual_modeling_host':socket.gethostname(), 'CPU_threads':8,
                'profile':config['profile'], 'upstream_official_rawprep_report':str(contract_path),
                'upstream_official_rawprep_report_sha256':sha(contract_path),
                'modeling_report':str(report_path), 'modeling_report_sha256':sha(report_path),
                'actual_upstream_EDDY':contract['actual_EDDY'],
                'completed_rawprep_sidecar':contract.get('completed_rawprep_sidecar'),
                'raw_frame_selection':contract.get('raw_frame_selection'),
                'restored_CPU_stage_lineage':contract.get('restored_CPU_stage_lineage'),
                'alignment_binding':contract.get('alignment_binding'),
                'harness_sha256':sha(__file__), 'files':outputs,
                'lineage':'Own official rawprep -> own official response -> own official MSMT-CSD -> own official mtnormalise. No FNIT frozen PT/response/FOD.'}
    save(contract_path_out, consumer)

    return report


def validate_case_binding(case, config, freeze, contract=None):
    """Each new namespace consumes an explicit case, producer, solver and raw-frame binding."""
    if case.get('binding_ready') is not True:
        raise ValueError('Case binding pending actual aligned upstream contract')
    if not case.get('ready_contract') or not case.get('upstream_runner_sha256'):
        raise ValueError('Explicit per-case report and producer SHA required')
    expected_selection = case.get('expected_selection')
    if not expected_selection or not {'ap_index','pa_index'} <= expected_selection.keys():
        raise ValueError('Frozen per-case AP/PA raw-frame selection required')
    if (freeze.get('manifest_sha256') != config['manifest_sha256'] or
            freeze.get('raw_root') != config['raw_root'] or
            case['upstream_runner_sha256'] not in freeze.get('source_sha256',{}).values()):
        raise ValueError('Per-case upstream source/raw freeze differs')
    if contract is not None:
        report = json.loads(Path(contract['files']['upstream_report']).read_text())
        if config.get('upstream_activation_format') == 'official_CPU_budget_reference_v1' and report.get('source_sha256') != freeze['source_sha256']:
            raise ValueError('Actual budget CPU report producer/helper SHA differs from activation')
        if report['subject'].removeprefix('sub-') != case['subject']:
            raise ValueError('Per-case subject differs')
        if any(report.get('selection',{}).get(key) != value for key,value in expected_selection.items()):
            raise ValueError('Per-case original raw-frame selection differs')
        if contract.get('actual_EDDY',{}).get('solver') != case.get('expected_upstream_solver'):
            raise ValueError('Per-case actual EDDY solver differs')
    return True


def read_bound_upstream_freeze(case, config):
    if config.get('upstream_activation_format') == 'official_CPU_budget_reference_v1':
        activation_path = Path(config['CPU_fallback_activation_report'])
        if sha(activation_path) != config['upstream_activation_sha256']:
            raise ValueError('Actual CPU budget activation SHA changed')
        activation = json.loads(activation_path.read_text())
        for path,digest in activation['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError('Actual CPU budget producer/helper/binary/evidence changed')
        if activation.get('CPU_threads_per_case')!=8 or activation.get('CPU_concurrency')!=2 or activation.get('CPU_stages_reexecuted') is not False or activation.get('scientific_EDDY_flags')!=EDDY_SCIENTIFIC_FLAGS:
            raise ValueError('Actual CPU budget activation parameters differ')
        return {'manifest_sha256':activation['manifest_sha256'],'raw_root':config['raw_root'],
                'source_sha256':activation['source_sha256'],'actual_activation_path':str(activation_path),
                'actual_activation_sha256':sha(activation_path),'raw_root_scope':'canonical raw verified independently per case'}
    return json.loads((Path(case['ready_contract']).parent.parent/'freeze.json').read_text())


def read_activation_marker(config):
    path = Path(config['CPU_fallback_activation_report'])
    actual = json.loads(path.read_text())
    if config.get('upstream_activation_format') != 'official_CPU_budget_reference_v1':
        return actual
    freeze = read_bound_upstream_freeze({},config)
    evidence_path = config['GPU_budget_evidence_path']
    if freeze['source_sha256'].get(evidence_path) != sha(evidence_path):
        raise ValueError('Actual budget failure evidence changed')
    evidence = json.loads(Path(evidence_path).read_text())
    return {'activated':True,'solver':'official_CPU','runner_sha256':freeze['source_sha256'][config['upstream_runner_path']],
            'activation_reason':'explicit_authorized_GPU_budget_stop_CPU_reference',
            'actual_activation_reason':actual['reason'],'actual_activation_path':str(path),
            'verified_failed_GPU_attempts':[{'report':item['report'],'sha256':item['report_sha256']} for item in evidence['subjects'].values()]}


def write_once_alignment_binding(path, evidence):
    """Create an immutable metadata proof; an existing proof must match every current digest."""
    path = Path(path)
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored != evidence:
            raise ValueError('Immutable pending alignment binding changed')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:
            stream.write(json.dumps(evidence,indent=2)+'\n')
    return {'path':str(path),'sha256':sha(path),'evidence':evidence}


def resolve_case_binding(case, config):
    if case.get('binding_ready') is True:
        return case, None
    if config.get('upstream_activation_format') != 'official_CPU_budget_reference_v1':
        return None, None
    subject = case['subject']
    expected_report = str(Path(config['upstream_producer_root'])/('sub-'+subject)/'report.json')
    if case.get('ready_contract') != expected_report or case.get('upstream_runner_sha256') != config['upstream_runner_sha256']:
        raise ValueError('Pending case must use the same explicit CPU budget producer')
    contract = ready_contract(expected_report,config['forbidden_input_roots'])
    if contract is None:
        return None, None
    freeze = read_bound_upstream_freeze(case,config)
    report = json.loads(Path(expected_report).read_text())
    lineage = contract.get('restored_CPU_stage_lineage')
    expected_source = str(Path(config['approved_restored_heldout_root'])/('sub-'+subject))
    expected_packing = str(Path(config['formal_FNIT_packing_root'])/('sub-'+subject)/'connectome/preproc/topup/B0_AP_PA.nii.gz')
    if (not lineage or lineage['source'] != expected_source or lineage['actual_FNIT_packing'] != expected_packing or
            not contract.get('completed_rawprep_sidecar',{}).get('verified_path')):
        raise ValueError('Pending alignment requires exact actual restored source, formal packing and verified sidecar')
    selection = {key:report.get('selection',{}).get(key) for key in ('ap_index','pa_index')}
    if any(type(value) is not int or value < 0 for value in selection.values()):
        raise ValueError('Pending actual raw-frame indices invalid')
    resolved = dict(case,binding_ready=True,expected_selection=selection)
    validate_case_binding(resolved,config,freeze,contract)
    validate_raw_binding(contract,config)
    sidecar = contract['completed_rawprep_sidecar']
    evidence = {'schema_version':1,'subject':subject,'binding_kind':'actual_verified_completed_CPU_budget_raw_frame_alignment',
                'actual_selection':selection,'upstream_report':expected_report,'upstream_report_sha256':sha(expected_report),
                'verified_completed_contract':sidecar['verified_path'],'verified_completed_contract_sha256':sidecar['verified_sha256'],
                'upstream_activation':config['CPU_fallback_activation_report'],'upstream_activation_sha256':config['upstream_activation_sha256'],
                'upstream_runner_sha256':case['upstream_runner_sha256'],'actual_source_sha256':report['source_sha256'],
                'canonical_raw_input_sha256':contract['canonical_raw_input_sha256'],
                'actual_formal_FNIT_packing':expected_packing,'actual_formal_FNIT_packing_sha256':sha(expected_packing),
                'own_raw_pair':str(Path(expected_report).parent/'topup/B0_AP_PA.nii.gz'),
                'own_raw_pair_sha256':sha(Path(expected_report).parent/'topup/B0_AP_PA.nii.gz'),
                'restored_CPU_source_report':lineage['report'],'restored_CPU_source_report_sha256':lineage['report_sha256'],
                'verification':'canonical raw frame arrays and affine equal to own pair and actual formal FNIT packing; exact actual solver/source/flags/nullUUID; all raw SHA verified',
                'GPU':False}
    proof = write_once_alignment_binding(Path(config['pending_alignment_binding_root'])/('sub-'+subject+'.alignment_binding.json'),evidence)
    return resolved, proof


def validate_activation_marker(marker, config):
    if (marker.get('activated') is not True or marker.get('solver') != 'official_CPU' or
            marker.get('runner_sha256') != config['upstream_runner_sha256'] or
            not marker.get('verified_failed_GPU_attempts')):
        raise ValueError('CPU reference activation proof does not meet authorized condition')
    policy = config.get('activation_policy','conditional_actual_GPU_allocation_failure')
    if policy == 'explicit_authorized_GPU_budget_stop_CPU_reference':
        if (config.get('explicit_CPU_budget_reference_authorization') is not True or
                marker.get('activation_reason') != policy):
            raise ValueError('Explicit budget-reference authorization and reason required')
    elif policy != 'conditional_actual_GPU_allocation_failure':
        raise ValueError('Unknown CPU activation policy')
    for attempt in marker['verified_failed_GPU_attempts']:
        if sha(attempt['report']) != attempt['sha256']:
            raise ValueError('Failed GPU attempt proof changed')
        if policy == 'explicit_authorized_GPU_budget_stop_CPU_reference':
            report = json.loads(Path(attempt['report']).read_text())
            stages = [stage for stage in report.get('commands',[]) if stage['stage']=='official_EDDY_GPU']
            if (report.get('completed') is not False or report.get('state')!='failed' or not stages or
                    stages[-1].get('returncode')!=-15 or stages[-1].get('budget_exceeded') is not True or
                    stages[-1].get('sampled_GPU_process_peak_bytes',0)<=20*1024**3):
                raise ValueError('Actual GPU budget stop proof required; do not label as OOM')
    return policy


def cpu_activation_preflight(config):
    """No new modeling namespace until actual completed independent CPU input."""
    marker_path = Path(config['CPU_fallback_activation_report'])
    if not marker_path.is_file():
        raise ValueError('CPU fallback activation has not occurred')
    marker = read_activation_marker(config)
    activation_policy = validate_activation_marker(marker,config)
    previous_status = Path(config['previous_modeling_root'])/'cohort_status.json'
    if previous_status.is_file():
        previous = json.loads(previous_status.read_text())
        selected = {case['subject'] for case in config['subjects'] if case.get('binding_ready') is True}
        if any(subject in selected and case.get('state') in ('running','completed') for subject,case in previous['subjects'].items()):
            raise ValueError('Previous GPU modeling running/completed; require per-case explicit binding, do not rerun')
    ready = []
    for declared_case in config['subjects']:
        case, alignment = resolve_case_binding(declared_case,config)
        if case is None:
            continue
        contract = ready_contract(case['ready_contract'],config['forbidden_input_roots'])
        if contract is None:
            continue
        if contract.get('actual_EDDY',{}).get('solver') != 'cpu':
            raise ValueError('CPU fallback binding contains a GPU-produced input')
        freeze = read_bound_upstream_freeze(case,config)
        validate_case_binding(case,config,freeze,contract)
        validate_raw_binding(contract,config)
        ready.append(case['subject'])
    if not ready:
        raise ValueError('No actual completed CPU contract; standby only, previous GPU namespace retired')
    return {'activation_report':str(marker_path),'activation_report_sha256':sha(marker_path),
            'completed_CPU_subjects_at_activation':ready,'activation_policy':activation_policy}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--poll-seconds', type=int, default=30); p.add_argument('--once', action='store_true')
    args = p.parse_args(); config = json.loads(Path(args.config).read_text())
    if config['profile']['kind'] != 'independent_official_fnit_mask_definitions' or not 1 <= len(config['subjects']) <= 10 or len({case['subject'] for case in config['subjects']}) != len(config['subjects']):
        raise ValueError('Expected frozen independent explicitly bound subject profile')
    if sha(config['manifest']) != config['manifest_sha256']:
        raise ValueError('Canonical manifest changed')
    activation_proof = cpu_activation_preflight(config)
    state_path = Path(config['output_root']) / 'cohort_status.json'; state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {'kind': 'independent_official_modeling_chain', 'actual_host': socket.gethostname(), 'threads': 8, 'subjects': {}, 'harness_sha256': sha(__file__), 'config_sha256': sha(args.config), 'CPU_fallback_activation_proof':activation_proof}
    while True:
        for case in config['subjects']:
            subject = case['subject']
            if state['subjects'].get(subject, {}).get('state') in ('completed', 'failed'):
                continue
            try:
                case, alignment = resolve_case_binding(case,config)
                if case is None:
                    state['subjects'][subject] = {'state':'waiting_true_verified_completed_alignment_binding'}; continue
                contract_path = case['ready_contract']
                freeze_path = Path(contract_path).parent.parent / 'freeze.json'
                if config.get('upstream_activation_format') != 'official_CPU_budget_reference_v1' and not freeze_path.is_file():
                    state['subjects'][subject] = {'state':'waiting_official_rawprep_namespace'}; continue
                freeze = read_bound_upstream_freeze(case,config)
                validate_case_binding(case,config,freeze)
                contract = ready_contract(contract_path, config['forbidden_input_roots'])
                if contract is None:
                    state['subjects'][subject] = {'state': 'waiting_official_rawprep'}; continue
                validate_case_binding(case,config,freeze,contract)
                contract = validate_raw_binding(contract, config)
                contract['alignment_binding'] = alignment
                if contract['subject'].removeprefix('sub-') != subject:
                    raise ValueError('Subject contract mismatch')
                state['subjects'][subject] = {'state': 'running'}; save(state_path, state)
                start = time.perf_counter(); result = model_case(config, subject, contract, contract_path)
                state['subjects'][subject] = {'state': 'completed', 'modeling_wall_s': time.perf_counter()-start, 'report': str(Path(config['output_root']) / ('sub-'+subject) / 'report.json'), 'consumer_contract':result['consumer_contract_path']}
            except Exception as error:
                state['subjects'][subject] = {'state': 'failed', 'error': repr(error)}
            save(state_path, state)
        state['state'] = 'completed' if all(v['state'] == 'completed' for v in state['subjects'].values()) and len(state['subjects']) == len(config['subjects']) else 'active_or_failed'
        save(state_path, state)
        if args.once or all(v['state'] in ('completed', 'failed') for v in state['subjects'].values()) and len(state['subjects']) == len(config['subjects']):
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
