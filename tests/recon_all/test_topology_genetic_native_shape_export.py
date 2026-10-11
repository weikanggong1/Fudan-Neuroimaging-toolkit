"""完整曲面诊断出口的来源与泄露拒绝门；小数组是合同，不是MRI基准。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TOOLS=Path(__file__).resolve().parents[2]/'tools'
sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location('native_shape_export',TOOLS/'export_recon_topology_genetic_repair_cache.py')
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)


class NativeShapeExportGates(unittest.TestCase):
    def _fixture(self,root):
        run=root/'run';source=root/'source';comparison=root/'comparison'
        (run/'candidate').mkdir(parents=True);(source/'tools').mkdir(parents=True)
        files={}
        for name in ('tools/compare_recon_genetic_mesh.py','validation/recon_all/accuracy_20261003/task_04/reference_surface_distance.py'):
            path=comparison/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture-not-a-benchmark\n')
            files[name]=dict(sha256=exporter.sha(path),bytes=path.stat().st_size)
        comparison_manifest=comparison/'SOURCE_MANIFEST.json';comparison_manifest.write_text(json.dumps(dict(files=files)))
        adapter=source/'tools/compare_recon_topology_genetic_repair_cache.py';adapter.write_bytes(b'fixture-adapter\n')
        manifest=source/'SOURCE_MANIFEST.json';manifest.write_text(json.dumps(dict(
            payload={str(adapter.relative_to(source)):dict(sha256=exporter.sha(adapter),size=adapter.stat().st_size)},
            comparison_manifest_sha256=exporter.sha(comparison_manifest))))
        arrays=dict(candidate_to_reference=np.array([0.,.01,.2,.4]),reference_to_candidate=np.array([0.,.02,.03,.4,.7]))
        checkpoint=run/'candidate/premesh.distances.private.npz';np.savez(checkpoint,**arrays)
        summaries={name+'_triangle':dict(count=len(value),mean_mm=float(value.mean()),p99_mm=float(np.quantile(value,.99)),max_mm=float(value.max()),over_0_1_mm=int(np.count_nonzero(value>.1)))for name,value in arrays.items()}
        topology=dict(vertices=4,faces=4,edges=6,euler=2,components=1,boundary_edges=0,nonmanifold_edges=0,degenerate_index_faces=0,duplicate_faces=0,valid=True)
        transforms={key:np.eye(4).tolist()for key in ('surface_ras_to_voxel','voxel_to_scanner_ras','surface_ras_to_scanner_ras')}
        transforms['nibabel_tkreg_f32_sha256']='b'*64
        result=dict(candidate_vertices=4,reference_vertices=5,candidate_faces=4,reference_faces=6,
            ordered_faces_equal=False,vertex_correspondence=False,raw_footer_geometry_f64_equal=False,
            raw_footer_geometry_f32_bits_equal=True,unit='scanner RAS mm',same_index_comparison='not_assessed',
            private_distance_checkpoint_sha256=exporter.sha(checkpoint),candidate_topology=topology,reference_topology=topology,
            raw_header_fields={key:dict(own=[1.,1.,1.],reference=[1.,1.,1.],f64_equal=True,f32_bits_equal=True,max_f64_difference=0.)for key in ('volume','voxelsize','xras','yras','zras','cras')},
            candidate_transforms=transforms,reference_transforms=transforms,
            candidate_sha256='a'*64,reference_sha256='b'*64,candidate_numeric_geometry_sha256='c'*64,reference_numeric_geometry_sha256='d'*64,**summaries)
        for key,count in (('forward_proof',4),('reverse_proof',5)):
            result[key]=dict(exact_zero_from_identical_used_target_vertex=0,full_triangle_search_vertices=count,total_vertices=count,all_vertices_evaluated=True)
        status='complete_readonly_all9_native_premesh_comparison'
        report=dict(status=status,result=result,script_sha256=exporter.sha(adapter),comparator_sha256=files['tools/compare_recon_genetic_mesh.py']['sha256'],
            distance_kernel_sha256=files['validation/recon_all/accuracy_20261003/task_04/reference_surface_distance.py']['sha256'],
            threads=4,cpu_affinity=[1,2,3,4],peak_rss_kib=1,reference_native_threads=1,reference_total_cpu_budget=4,
            reference_cpu_affinity=[5,6,7,8],reference_seed=1234,reference_source_commit='a'*40,reference_wrapper_source_commit='b'*40,
            input_sha256=dict(nofix='a'*64,qsphere='b'*64,brain='c'*64))
        for key in ('own_source_manifest_sha256','own_cache_report_sha256','own_report_sha256','own_complete_sha256','own_exit_sha256',
            'public_metrics_sha256','reference_report_sha256','reference_complete_sha256','reference_exit_sha256','reference_program_sha256'):
            report[key]='a'*64
        (run/'EXIT.json').write_text(json.dumps(dict(exit_code=0,complete_present=True,complete_diagnostic_wall_seconds=1.25)))
        args=dict(input_run=run,frozen_source=source,source_manifest=manifest,expected_source_manifest_sha256=exporter.sha(manifest),
            expected_exit_sha256=exporter.sha(run/'EXIT.json'),comparison_source=comparison,comparison_manifest=comparison_manifest,
            expected_comparison_manifest_sha256=exporter.sha(comparison_manifest),output_directory=root/'public')
        self._seal(args,report)
        return args,report

    def _seal(self,args,report):
        run=args['input_run'];(run/'candidate/report.private.json').write_text(json.dumps(report))
        args['expected_report_sha256']=exporter.sha(run/'candidate/report.private.json')
        (run/'candidate/COMPLETE.json').write_text(json.dumps(dict(status=report['status'],report_sha256=args['expected_report_sha256'])))
        args['expected_complete_sha256']=exporter.sha(run/'candidate/COMPLETE.json')

    def _reject(self,change,message):
        with tempfile.TemporaryDirectory()as directory:
            args,report=self._fixture(Path(directory));change(args,report);self._seal(args,report)
            with self.assertRaisesRegex((ValueError,TypeError),message):exporter.export_completed_native_shape(**args)
            self.assertFalse(args['output_directory'].exists())

    def test_full_array_stats_and_numeric_types_are_preserved(self):
        with tempfile.TemporaryDirectory()as directory:
            args,report=self._fixture(Path(directory));exporter.export_completed_native_shape(**args)
            metrics=json.loads((args['output_directory']/'metrics.json').read_text())
            self.assertEqual(metrics['candidate_to_reference_triangle'],report['result']['candidate_to_reference_triangle'])
            self.assertIs(type(metrics['threads']),int);self.assertIs(type(metrics['ordered_faces_equal']),bool)

    def test_rehashed_incorrect_summary_cannot_replace_complete_array_result(self):
        self._reject(lambda a,r:r['result']['candidate_to_reference_triangle'].update(max_mm=.1),'summary differs')

    def test_rehashed_private_header_dictionary_key_is_rejected(self):
        self._reject(lambda a,r:r['result']['raw_header_fields']['volume'].update(own={'/private/subject/path':1}),'three-value list')

    def test_boolean_matrix_element_is_not_a_coordinate(self):
        self._reject(lambda a,r:r['result']['candidate_transforms']['surface_ras_to_voxel'][0].__setitem__(0,True),'numeric scalar')

    def test_mutated_comparator_payload_is_rejected(self):
        self._reject(lambda a,r:(a['comparison_source']/'tools/compare_recon_genetic_mesh.py').write_bytes(b'changed'),'payload changed')

    def test_incomplete_all_vertex_proof_is_rejected(self):
        self._reject(lambda a,r:r['result']['reverse_proof'].update(full_triangle_search_vertices=4),'all-vertex')

    def test_negative_wall_time_is_rejected_before_public_write(self):
        def change(args,report):
            path=args['input_run']/'EXIT.json';record=json.loads(path.read_text());record['complete_diagnostic_wall_seconds']=-1.
            path.write_text(json.dumps(record));args['expected_exit_sha256']=exporter.sha(path)
        self._reject(change,'nonnegative diagnostic')


if __name__=='__main__':unittest.main()
