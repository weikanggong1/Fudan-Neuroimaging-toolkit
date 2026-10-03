"""检查自产sphere真正进入完整注册，另列文件SHA与几何对应关系。"""
import argparse, importlib.util, json, pathlib, sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-root', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    script = pathlib.Path(__file__).with_name('compare_stages.py')
    spec = importlib.util.spec_from_file_location('_geometry_stage_compare', script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = []
    for subject in ('sub01', 'sub02'):
        for hemi in ('lh', 'rh'):
            chain = args.task_root/'chains'/subject/hemi
            if not (chain/'output/report.json').exists():
                continue
            provenance = json.loads((chain/'provenance.json').read_text())
            sphere = json.loads((args.task_root/'cold_pairs'/subject/hemi/'sphere/candidate/report.json').read_text())
            registration = json.loads((chain/'output/report.json').read_text())
            reference = args.task_root/'cold_pairs'/subject/hemi/'register/candidate'
            if not (reference/'report.json').exists():
                continue
            row = module.compare(reference, chain/'output', 'register', hemi, False)
            row.update(subject=subject, kind='candidate continuous sphere to register', sphere_producer_commit=provenance['sphere_source_commit'], sphere_producer_output_sha256=sphere['output_sha256'], consumed_sphere_sha256=registration['input_sha256']['sphere'])
            row['sphere_provenance_verified']=sphere['status']=='complete' and sphere['output_sha256']==provenance['input_sha256']['sphere']==registration['input_sha256']['sphere']
            row['strict_chain_propagation']=row['sphere_provenance_verified'] and all(row[key] for key in ('coordinate_array_equal','coordinate_float32_bits_equal','geometry_footer_equal','trajectory_equal'))
            rows.append(row)
    args.output.write_text(json.dumps({'rows':rows, 'required_chains':4, 'complete_chain_count':len(rows), 'overall_equivalence':'not_assessed', 'scope':'actual new sphere consumed by complete two-pass registration; comparison to same frozen-input candidate registration; whole T1 and regional metrics remain coordinator acceptance'}, indent=2)+'\n')
    print(json.dumps([{key:row[key] for key in ('subject','hemisphere','sphere_provenance_verified','strict_chain_propagation')} for row in rows], indent=2))


if __name__ == '__main__':
    main()
