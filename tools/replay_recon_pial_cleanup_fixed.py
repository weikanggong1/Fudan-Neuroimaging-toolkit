"""只回放自产接受末态的pin/完整清理，比较实际冻结约束；不生成生产pial。"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import nibabel as nib
from fnit.recon_all.place_surface_final_cleanup import pin_medial_wall,pial_cleanup_fixed_vertices,pial_cleanup_mask,repair_intersections
from fnit.recon_all.place_surface_quality import proper_pairs,mesh_structure,placement_structure_passed
from fnit.recon_all.thread_budget import thread_budget


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def write(path,data):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True);parser.add_argument('--hemisphere',choices=['lh','rh'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0');parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--cleanup-constraint-mode',choices=['source_rip','cortex_pinned_union'],default='source_rip')
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    h=a.hemisphere;observation=a.case/'pial-observation';events=json.loads((observation/'events.json').read_text())
    step=next(event for event in reversed(events) if event['kind']=='step');cleanup=next(event for event in reversed(events) if event['kind']=='cleanup')
    with np.load(observation/step['file']) as arrays:accepted=arrays['accepted'].copy();ripped=arrays['ripped'].astype(bool)
    with np.load(observation/cleanup['file']) as arrays:old=arrays['after'].copy()
    white,faces=nib.freesurfer.read_geometry(str(a.case/f'candidate/surf/{h}.white'));white=white.astype(np.float32);faces=faces.astype(np.int32)
    cortex=nib.freesurfer.read_label(str(a.case/f'candidate/label/{h}.cortex.label'))
    pin_fixed=pial_cleanup_fixed_vertices(ripped=ripped,cortex_vertices=cortex)
    fixed=pial_cleanup_mask(ripped=ripped,cortex_vertices=cortex,constraint_mode=a.cleanup_constraint_mode)
    pinned=pin_medial_wall(accepted,white,cortex)
    report=dict(schema_version=1,scope='own accepted terminal replay only; not a new four-pass run or production output',status='running',
        cleanup_constraint_mode=a.cleanup_constraint_mode,
        soap_neighbor_mask='original_rip' if a.cleanup_constraint_mode=='source_rip' else 'original_rip_union_outside_cortex',
        optimizer_and_labels_unchanged=True,own_accepted_sha256=digest(accepted),last_step=step['step'],last_pass=step['pass_index'],
        frozen_event_sha256=sha(observation/'events.json'),script_sha256=sha(__file__),
        source_sha256={name:sha(Path(__file__).parents[1]/'src/fnit/recon_all'/name) for name in ['place_surface_final_cleanup.py','place_surface_quality.py','place_pial_python.py']},
        original_ripped_vertices=int(ripped.sum()),cleanup_fixed_vertices=int(fixed.sum()),additional_pinned_vertices=int((fixed&~ripped).sum()),
        baseline_fixed_different_elements_to_white=int(np.count_nonzero(old[fixed]!=white[fixed])),
        baseline_union_pinned_different_elements_to_white=int(np.count_nonzero(old[pin_fixed]!=white[pin_fixed])),
        no_standard_pial_surface_written=True,overall_equivalence='not_assessed',operator_threshold='not_assessed')
    path=a.output/'report.json';write(path,report)
    with thread_budget(threads=a.threads):
        import torch
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        if a.device.startswith('cuda:'):torch.cuda.synchronize(a.device)
        tick=time.perf_counter()
        candidate,diagnostics=repair_intersections(pinned,faces,fixed,marking_backend='source_torch' if a.device.startswith('cuda:') else 'source_numba',
            device=a.device,candidate_grid_cells_per_axis=3 if a.device.startswith('cuda:') else 2)
        if a.device.startswith('cuda:'):torch.cuda.synchronize(a.device)
        report.update(cleanup_seconds=time.perf_counter()-tick,cleanup=diagnostics,
            candidate_fixed_different_elements_to_white=int(np.count_nonzero(candidate[fixed]!=white[fixed])),
            candidate_union_pinned_different_elements_to_white=int(np.count_nonzero(candidate[pin_fixed]!=white[pin_fixed])),
            candidate_different_elements_to_baseline=int(np.count_nonzero(candidate!=old)),candidate_sha256=digest(candidate))
        np.save(a.output/'candidate_cleanup.private.npy',candidate);write(path,report)
        for label,xyz in [('baseline',old),('candidate',candidate)]:
            _,self_report=proper_pairs(first_vertices=xyz,faces=faces)
            _,cross_report=proper_pairs(first_vertices=xyz,second_vertices=white,faces=faces)
            report[label+'_geometry']=dict(self_proper=self_report,white_pial_proper=cross_report,structure=mesh_structure(vertices=xyz,faces=faces));write(path,report)
        report['quality_gates']=dict(source_marker_zero=diagnostics['intersecting_faces_after']==0,
            selected_mode_fixed_coordinates_unchanged=report['candidate_fixed_different_elements_to_white']==0,
            strict_self_zero=report['candidate_geometry']['self_proper']['proper_pairs']==0,
            strict_white_pial_zero=report['candidate_geometry']['white_pial_proper']['proper_pairs']==0,
            closed_single_component_nondegenerate=placement_structure_passed(report['candidate_geometry']['structure']))
        report['quality_passed']=all(report['quality_gates'].values())
        report['status']='complete_replay_quality_passed' if report['quality_passed'] else 'complete_replay_quality_failed'
        write(path,report)
if __name__=='__main__':main()
