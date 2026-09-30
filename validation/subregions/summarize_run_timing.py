"""Summarize completed and in-progress unified GEMS run logs.

Whole-run and structure totals contain the EM/mesh child timers. Keep these
levels separate: adding child timers to their parents would count work twice.
Unattributed elapsed time includes unlogged preparation, output and pipeline work.
"""

import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import re


_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) (.*)$")
_STAGE = re.compile(r"^(\S+) (segmentation|intensity) stage (\d+)/(\d+): sigma=(\S+) iterations=(\d+)$")
_INITIAL = re.compile(r"^GEMS initial EM: ([\d.]+) s, (\d+) valid voxels$")
_OUTER = re.compile(r"^GEMS outer (\d+)/(\d+) (EM|mesh): ([\d.]+) s(?:, (\d+) evaluations)?$")
_FINISHED = re.compile(r"^Finished (\S+): (.*)$")


def summarize_log(path):
    structures = {}
    current_stage = None
    run_summary = None
    timestamps = []
    parse_errors = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        match = _LINE.match(line)
        if match is None:
            if line.startswith('{'):
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and 'mode' in candidate and 'seconds' in candidate:
                    run_summary = candidate
            continue
        timestamp, message = match.groups()
        timestamps.append(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S,%f'))
        if message.startswith('Starting '):
            name = message.removeprefix('Starting ')
            structures.setdefault(name, {'complete': False, 'intensity_stages': []})
            current_stage = None
            continue
        match = _STAGE.match(message)
        if match is not None:
            name, phase, index, count, sigma, iterations = match.groups()
            if current_stage is not None:
                current_stage['complete'] = True
            current_stage = None
            if phase == 'intensity':
                current_stage = {'index': int(index), 'stage_count': int(count),
                                 'sigma': float(sigma), 'outer_limit': int(iterations),
                                 'complete': False, 'initial_em_seconds': 0., 'outers': {}}
                structures[name]['intensity_stages'].append(current_stage)
            continue
        match = _FINISHED.match(message)
        if match is not None:
            name, text = match.groups()
            try:
                report = ast.literal_eval(text)
            except (ValueError, SyntaxError) as error:
                parse_errors.append({'line': number, 'error': str(error)})
                continue
            structure = structures[name]
            structure['complete'] = True
            components = report.get('timing_seconds', {})
            structure['reported_total_seconds'] = components.get('total', report.get('seconds'))
            structure['reported_primary_components'] = {k: v for k, v in components.items() if k != 'total'}
            structure['reported_segmentation_fit_seconds'] = report.get('segmentation_fit', {}).get('seconds')
            structure['peak_gpu_gib'] = report.get('peak_gpu_gib')
            if current_stage is not None:
                current_stage['complete'] = True
                current_stage = None
            continue
        if current_stage is None:
            continue
        match = _INITIAL.match(message)
        if match is not None:
            seconds, voxels = match.groups()
            current_stage['initial_em_seconds'] = float(seconds)
            current_stage['valid_voxels'] = int(voxels)
            continue
        match = _OUTER.match(message)
        if match is not None:
            outer, limit, phase, seconds, evaluations = match.groups()
            entry = current_stage['outers'].setdefault(int(outer), {'index': int(outer)})
            entry[phase.lower() + '_seconds'] = float(seconds)
            if evaluations is not None:
                entry['evaluations'] = int(evaluations)

    for structure in structures.values():
        for stage in structure['intensity_stages']:
            stage['outers'] = [stage['outers'][i] for i in sorted(stage['outers'])]
            stage['mesh_calls'] = sum('mesh_seconds' in entry for entry in stage['outers'])
            stage['mesh_seconds'] = round(sum(entry.get('mesh_seconds', 0.) for entry in stage['outers']), 2)
            stage['outer_em_seconds'] = round(sum(entry.get('em_seconds', 0.) for entry in stage['outers']), 2)
            stage['evaluations'] = sum(entry.get('evaluations', 0) for entry in stage['outers'])
            stage['seconds_per_evaluation'] = (stage['mesh_seconds'] / stage['evaluations']
                                               if stage['evaluations'] else None)
        stages = structure['intensity_stages']
        structure['logged_intensity_mesh_seconds'] = round(sum(s['mesh_seconds'] for s in stages), 2)
        structure['logged_intensity_em_seconds'] = round(sum(s['initial_em_seconds'] + s['outer_em_seconds']
                                                             for s in stages), 2)
        structure['logged_mesh_evaluations'] = sum(s['evaluations'] for s in stages)
        structure['completed_intensity_stages'] = sum(s['complete'] for s in stages)
        if structure['complete'] and not structure['reported_primary_components']:
            structure['unattributed_structure_seconds'] = (
                structure['reported_total_seconds'] - (structure['reported_segmentation_fit_seconds'] or 0.)
                - structure['logged_intensity_em_seconds'] - structure['logged_intensity_mesh_seconds'])

    completed_seconds = sum(s.get('reported_total_seconds', 0.) for s in structures.values())
    return {'log': str(Path(path).resolve()), 'run_complete': run_summary is not None,
            'run_summary': run_summary,
            'observed_log_span_seconds': ((timestamps[-1] - timestamps[0]).total_seconds()
                                          if timestamps else None),
            'reported_completed_structure_seconds': completed_seconds,
            'logged_intensity_em_seconds': round(sum(s['logged_intensity_em_seconds']
                                                    for s in structures.values()), 2),
            'logged_intensity_mesh_seconds': round(sum(s['logged_intensity_mesh_seconds']
                                                      for s in structures.values()), 2),
            'logged_mesh_evaluations': sum(s['logged_mesh_evaluations'] for s in structures.values()),
            'unattributed_run_seconds': (run_summary['seconds'] - completed_seconds
                                         if run_summary is not None else None),
            'structures': structures, 'parse_errors': parse_errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps({'logs': [summarize_log(path) for path in args.logs]}, indent=2))


if __name__ == '__main__':
    main()
