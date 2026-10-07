"""Bind existing real inputs, atlas files and frozen source before CPU recipes."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = Path('/cwStorage/home/gongwk/Notebook_code/FNIT')
    workspace = base / 'workspaces/smri_cpu_20261004/remaining_20261004/gems/cpu-epsilon-v1'
    run = base / 'runs/smri_cpu_20261004/remaining_20261004/gems/cpu-epsilon-v1'
    old_source = base / 'workspaces/smri_cpu_20261004/remaining_20261004/gems/baseline-f1cbdab/src'
    new_source = workspace / 'source/src'
    subject = base / 'legacy/freesurfer_synth/work/fnit_subregions_unified_20260930/ten_public_t1_20261002/cases/sub-02/official/subjects/sub-02/mri'
    atlas = base / 'legacy/freesurfer_synth/work/fnit_subregions_unified_20260930/atlases'
    expected = {'norm.mgz': '001021a47f102bb9772f7f736b65f74740334e86dfd1afd0386ca250cb25b698',
                'aseg.mgz': 'ea02b3cd278c8eb229a4cd6a5bec982a0a6175f7d34ab3f651696902e1a259f8',
                'wmparc.mgz': 'bd45c6371c7d389cfc5d9ce5cabe321e362d32cce6feb35fdaea81e7606a469c'}
    for name, checksum in expected.items():
        if digest(subject / name) != checksum:
            raise RuntimeError('fixed real input changed: ' + name)
    for structure in ('thalamus', 'hippo-amygdala-left', 'hippo-amygdala-right'):
        first = json.loads((run / ('first-' + structure + '.public.json')).read_text())
        if not first['prototype_gate_passed']:
            raise RuntimeError('actual first-state gate failed')
    frozen = json.loads((workspace / 'source.public.json').read_text())
    for name, checksum in frozen['gems_source_files'].items():
        if digest(new_source / name) != checksum:
            raise RuntimeError('candidate frozen GEMS source changed: ' + name)
    files = {str(p.relative_to(atlas)): digest(p) for p in sorted(atlas.rglob('*')) if p.is_file()}
    # Both arms use the same packed atlas. Installed source files establish the
    # link to the first-state oracle; prepared pack helper files are also bound.
    upstream = Path('/public/software/apps/Freesurfer/8.2.0-1/average')
    upstream_dirs = {'brainstem': upstream / 'BrainstemSS/atlas',
                     'thalamus': upstream / 'ThalamicNuclei/atlas',
                     'hippo-amygdala-left': upstream / 'HippoSF/atlas',
                     'hippo-amygdala-right': upstream / 'HippoSF/atlas'}
    equality = {}
    for family, origin in upstream_dirs.items():
        for name in ('AtlasMesh.gz', 'AtlasDump.mgz', 'compressionLookupTable.txt'):
            packed = atlas / family / name
            original = origin / name
            equality[family + '/' + name] = {'pack_sha256': digest(packed),
                                           'installed_sha256': digest(original),
                                           'exact': digest(packed) == digest(original)}
            if not equality[family + '/' + name]['exact']:
                raise RuntimeError('oracle atlas differs from recipe pack: ' + family + '/' + name)
    binding = {'schema': 'fnit.gems.cpu-epsilon.recipes.binding.v1',
               'input_sha256': expected, 'atlas_files_sha256': files,
               'atlas_original_file_comparison': equality,
               'sources': {tag: {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}
                           for tag, root in [('baseline', old_source), ('candidate', new_source)]},
               'helpers_sha256': {name: digest(workspace / name) for name in
                                  ('worker.py', 'gems_stage_contract.py', 'queue_runner_v3.py')},
               'baseline_brainstem_reuse': str(base / 'runs/smri_cpu_20261004/remaining_20261004/gems/nodecw7-cache-v1/baseline-f1cbdab/artifacts'),
               'accepted_gaussian_repair': 'candidate includes cafbfa04; baseline lacks the no-hyper broadcast fix, but all real recipe EM calls provide hyperparameters and synthetic calls use fixed Gaussians',
               'reference_labels_used_for_fitting': False,
               'timing_scope': 'cold processes for the fixed norm/aseg/wmparc recipe stage; no raw T1 reconstruction; reused official arrays do not provide a same-node official speed denominator'}
    (run / 'recipe_binding.public.json').write_text(json.dumps(binding, indent=2) + '\n')
    jobs = []
    # Separate family calls provide complete fitted outputs early. The HA call
    # executes both hemispheres with the ordinary public API and shared context.
    for tag, family, source in [('candidate', 'brainstem', new_source),
                               ('baseline', 'thalamus', old_source),
                               ('candidate', 'thalamus', new_source),
                               ('baseline', 'hippo-amygdala', old_source),
                               ('candidate', 'hippo-amygdala', new_source)]:
        identifier = tag + '-' + family
        output = run / 'recipes' / identifier / 'artifacts'
        command = [str(base / 'envs/default/bin/python'), str(workspace / 'gems_stage_contract.py'),
                   '--worker', str(workspace / 'worker.py'), 'subregions',
                   '--t1', str(subject / 'norm.mgz'), '--aseg', str(subject / 'aseg.mgz'),
                   '--wmparc', str(subject / 'wmparc.mgz'), '--atlas-root', str(atlas),
                   '--output-dir', str(output), '--device', 'cpu', '--threads', '8',
                   '--structures', family, '--optimization', 'fast', '--save-posteriors']
        jobs.append({'id': identifier, 'argv': command, 'env': {'PYTHONPATH': str(source)},
                     'timeout_seconds': 18000, 'expected_outputs': [str(output / name) for name in
                       ('worker.json', 'report.json', 'subregions_native.nii.gz', 'volumes.tsv',
                        'fit_parameters.private.npz', 'fit_contract.private.json', 'stage_contract.public.json')]})
    plan = {'schema': 'fnit.gems.cpu-epsilon.recipe-plan.v1', 'binding_sha256': digest(run / 'recipe_binding.public.json'), 'jobs': jobs}
    (workspace / 'recipe_jobs.private.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'jobs': [job['id'] for job in jobs], 'atlas_files': len(files),
                      'source_files': {key: len(value) for key, value in binding['sources'].items()}}))


if __name__ == '__main__':
    main()
