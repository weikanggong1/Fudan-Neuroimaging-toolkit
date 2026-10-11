"""完整自产九缺陷末态脑图和计时；输入脑MRI留在服务器，图中位移不是原版误差。"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())


def plot_native_shape(*,shape_run,shape_source,shape_source_manifest_sha256,
        publication_directory,expected_metrics_sha256,candidate_surface,
        reference_surface,brain,output_directory):
    """共同scannerRAS的完整曲面差异叠加MRI；各自头变换，不拟合或重排序。"""
    import importlib.util
    import time
    import numpy as np
    import nibabel as nib
    import nibabel.freesurfer.io as fsio
    D,S,U,out=map(Path,(shape_run,shape_source,publication_directory,output_directory))
    if out.exists():raise FileExistsError('new native-shape figure directory required')
    tick=time.perf_counter();metrics=load(U/'metrics.json');receipt=load(U/'EXPORT_RECEIPT.json')
    if (sha(U/'metrics.json')!=expected_metrics_sha256
            or receipt['payload']['metrics']['sha256']!=expected_metrics_sha256
            or receipt['actual_private_report_sha256']!=sha(D/'candidate/report.private.json')
            or receipt['actual_complete_sha256']!=sha(D/'candidate/COMPLETE.json')
            or receipt['actual_exit_sha256']!=sha(D/'EXIT.json')
            or load(D/'EXIT.json')['exit_code']!=0):raise ValueError('same completed diagnostic/export required')
    manifest=S/'SOURCE_MANIFEST.json'
    if sha(manifest)!=shape_source_manifest_sha256 or metrics['source_manifest_sha256']!=shape_source_manifest_sha256:
        raise ValueError('actual scanner-space adapter manifest required')
    mf=load(manifest)
    for name,item in mf['payload'].items():
        if sha(S/name)!=item['sha256']or(S/name).stat().st_size!=item['size']:raise ValueError('shape source payload changed')
    path=S/'tools/compare_recon_topology_genetic_repair_cache.py'
    if sha(path)!=metrics['sha256']['script_sha256']:raise ValueError('actual adapter SHA required')
    spec=importlib.util.spec_from_file_location('_frozen_native_shape_adapter',path)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    surfaces=[Path(candidate_surface),Path(reference_surface)]
    if ([sha(p)for p in surfaces]!=[metrics['sha256']['candidate_sha256'],metrics['sha256']['reference_sha256']]
            or sha(brain)!=metrics['input_sha256']['brain']):raise ValueError('actual complete meshes/brain differ')
    checkpoint=D/'candidate/premesh.distances.private.npz'
    if sha(checkpoint)!=metrics['sha256']['private_distance_checkpoint_sha256']:raise ValueError('actual full distance arrays required')
    image=nib.load(brain);data=np.asarray(image.dataobj);scanner_to_voxel=np.linalg.inv(image.affine)
    fields=[]
    with np.load(checkpoint,allow_pickle=False)as arrays:
        distances=[arrays[key].copy()for key in ('candidate_to_reference','reference_to_candidate')]
    for domain,surface,distance in zip(('candidate','reference'),surfaces,distances):
        vertices,faces,geometry=fsio.read_geometry(str(surface),read_metadata=True)
        transforms=adapter.scanner_transform(geometry)
        for key in ('surface_ras_to_voxel','voxel_to_scanner_ras','surface_ras_to_scanner_ras'):
            if not np.array_equal(transforms[key],metrics['actual_transforms'][domain][key]):raise ValueError('actual numeric transform differs')
        matrix=np.asarray(transforms['surface_ras_to_scanner_ras']);scanner=vertices@matrix[:3,:3].T+matrix[:3,3]
        vox=scanner@scanner_to_voxel[:3,:3].T+scanner_to_voxel[:3,3]
        if distance.shape!=(len(vertices),)or distance.dtype!=np.float64 or not np.isfinite(distance).all()or np.any(distance<0):raise ValueError('all finite full-mesh distances required')
        if not np.isfinite(vox).all():raise ValueError('finite brain transform required')
        fields.append((vox,np.asarray(faces),scanner))
    maximum=int(np.argmax(distances[1]));location=fields[1][0][maximum]
    slices=[int(np.clip(np.rint(location[axis]),0,data.shape[axis]-1))for axis in range(3)]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    colors=['#ff9d57','#51c7ed'];labels=['own premesh','fixed native premesh']
    fig,axes=plt.subplots(2,3,figsize=(13,8),dpi=160);segments_record=[];near_counts=[]
    vmax=max(float(d.max())for d in distances)
    for axis in range(3):
        remaining=[a for a in range(3)if a!=axis];plane=slices[axis];counts=[];near=[]
        for row in range(2):
            axes[row,axis].imshow(np.take(data,plane,axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=180)
            axes[row,axis].set_xlim(0,data.shape[remaining[0]]);axes[row,axis].set_ylim(0,data.shape[remaining[1]]);axes[row,axis].set_axis_off()
        for (vox,faces,scanner),distance,color,label in zip(fields,distances,colors,labels):
            triangles=vox[faces];signed=triangles[:,:,axis]-plane;grouped={}
            for a,b in ((0,1),(1,2),(2,0)):
                mask=((signed[:,a]<=0)&(signed[:,b]>0))|((signed[:,b]<=0)&(signed[:,a]>0))
                ids=np.flatnonzero(mask);alpha=-signed[ids,a]/(signed[ids,b]-signed[ids,a]);points=triangles[ids,a]+alpha[:,None]*(triangles[ids,b]-triangles[ids,a])
                for face,point in zip(ids,points):grouped.setdefault(int(face),[]).append(point[remaining])
            segments=np.asarray([points for points in grouped.values()if len(points)==2]);counts.append(len(segments))
            if len(segments):axes[0,axis].add_collection(LineCollection(segments,colors=color,linewidths=.65,label=label))
            selected=(np.abs(vox[:,axis]-plane)<=1.5)&(distance>.1);near.append(int(np.count_nonzero(selected)))
            axes[1,axis].scatter(vox[selected,remaining[0]],vox[selected,remaining[1]],c=distance[selected],cmap='magma',vmin=0,vmax=vmax,s=7,alpha=.9)
        segments_record.append(counts);near_counts.append(near)
        axes[0,axis].plot(location[remaining[0]],location[remaining[1]],'+',color='red',markersize=9)
        axes[0,axis].set_title(['sagittal','coronal','axial'][axis]+' / voxel '+str(plane),fontsize=9)
    axes[0,0].legend(loc='lower left',fontsize=8,facecolor='black',labelcolor='white')
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,vmax),cmap='magma'),ax=axes[1,:].tolist(),label='point to complete opposite triangle mesh / mm',shrink=.7)
    fig.suptitle('Common scanner RAS, no fitting: slices at native-to-own maximum '+format(float(distances[1][maximum]),'.3f')+' mm; lower panels show >0.1 mm points',fontsize=10)
    out.mkdir(mode=0o700,parents=True);fig.savefig(out/'native_shape_t1.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),dpi=160)
    for row,((vox,faces,scanner),distance,label)in enumerate(zip(fields,distances,labels)):
        selected=distance>.1
        for axis in range(3):
            remaining=[a for a in range(3)if a!=axis];ax=axes[row,axis]
            ax.scatter(scanner[:,remaining[0]],scanner[:,remaining[1]],s=.15,color='gray',alpha=.08)
            ax.scatter(scanner[selected,remaining[0]],scanner[selected,remaining[1]],s=5,c=distance[selected],cmap='magma',vmin=0,vmax=vmax)
            ax.set_aspect('equal');ax.set_title(label+' / '+['YZ','XZ','XY'][axis]+' scanner RAS',fontsize=9);ax.set_xlabel('mm');ax.set_ylabel('mm')
    fig.suptitle('All vertices evaluated against complete triangle meshes; colored points >0.1 mm (display threshold)',fontsize=10)
    fig.tight_layout();fig.savefig(out/'native_shape_error_projection.png');plt.close(fig)
    report=dict(status='complete_actual_native_shape_figures',tool_sha256=sha(__file__),source_manifest_sha256=shape_source_manifest_sha256,
        adapter_sha256=sha(path),metrics_sha256=expected_metrics_sha256,brain_sha256=sha(brain),distance_checkpoint_sha256=sha(checkpoint),
        surface_sha256=[sha(p)for p in surfaces],slice_voxel_indices=slices,contour_segments=segments_record,near_slice_over_0_1_mm=near_counts,
        native_max_vertex=maximum,native_max_scanner_ras_mm=fields[1][2][maximum].tolist(),native_max_brain_voxel=location.tolist(),
        brain_voxel_to_scanner_ras=image.affine.tolist(),diagnostic_plot_seconds=time.perf_counter()-tick,coordinate_fitting_performed=False,
        mesh_reordered=False,interpretation='posthoc native-reference full-triangle distance; no corresponding vertex subtraction',
        whole_recon_all='not_assessed',overall_metric_equivalence='not_assessed',payload={p.name:dict(sha256=sha(p),size=p.stat().st_size)for p in out.glob('*.png')})
    (out/'PLOT_RECEIPT.json').write_text(json.dumps(report,indent=2)+chr(10));return report


def plot_completed(*,frozen_source,source_manifest_sha256,publication_run,premesh_run,orig_run,
        nofix,brain,output_directory):
    """具名路径输入，验证来源和全部三表面/原MRI SHA，再画同平面的真实脑图。

    publication_run含已安全导出的premesh/orig metrics+EXPORT_RECEIPT。
    nofix为原三角面surface RAS/mm，brain为同链uint8 MRI。premesh_run
    和orig_run为完成producer目录，surface仍私有。本函数仅事后读取，
    输出两PNG和含SHA/平面/数字的PLOT_RECEIPT.json，不输出原始MRI/网格。
    切片取自产orig体素坐标的中位数；所有面与该平面相交的线段都画出。
    面和顶点域不同，不逐顶点相减，不画成官方误差。
    """
    S,U,P,O=map(Path,(frozen_source,publication_run,premesh_run,orig_run));out=Path(output_directory)
    if out.exists():raise FileExistsError('new figure directory required')
    manifest=S/'SOURCE_MANIFEST.private.json'
    if sha(manifest)!=source_manifest_sha256:raise ValueError('actual completed source manifest required')
    mf=load(manifest)
    for n,v in mf['payload'].items():
        p=S/n
        if sha(p)!=v['sha256']or p.stat().st_size!=v['size']:raise ValueError('immutable plot source changed')
    records={}
    for stage in ('premesh','orig'):
        report=U/stage/'metrics.json';receipt=load(U/stage/'EXPORT_RECEIPT.json')
        if receipt['source_manifest_sha256']!=source_manifest_sha256 or receipt['payload']['metrics']['sha256']!=sha(report):raise ValueError('safe metrics binding differs')
        records[stage]=load(report)
    prem,orig=records['premesh'],records['orig']
    if prem['completed_defects']!=9 or orig['status']!='complete_orig_stage':raise ValueError('completed all-nine/orig required')
    surfaces=[Path(nofix),P/'candidate/lh.orig.premesh',O/'candidate/lh.orig']
    expected=[prem['input_sha256']['nofix'],prem['premesh_sha256'],orig['orig_sha256']]
    if any(sha(p)!=v for p,v in zip(surfaces,expected))or sha(brain)!=prem['input_sha256']['brain']:raise ValueError('actual brain/surface bytes differ')
    sys.path[:0]=[str(S/'src')]
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(S/'src/fnit/recon_all'))
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    module=importlib.import_module('fnit.recon_all.topology_edge_mri_score')
    if Path(module.__file__).resolve()!=S/'src/fnit/recon_all/topology_edge_mri_score.py' or sha(module.__file__)!=mf['payload']['src/fnit/recon_all/topology_edge_mri_score.py']['sha256']:raise ValueError('actual sampler module binding differs')
    import numpy as np
    import nibabel as nib
    import nibabel.freesurfer.io as fsio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    image=nib.load(brain);data=np.asarray(image.dataobj)
    matrix=source_topology_ras_to_voxel(volume_header=image.header)
    fields=[]
    for p in surfaces:
        vertices,faces=fsio.read_geometry(str(p));vox=np.column_stack((vertices,np.ones(len(vertices))))@np.asarray(matrix).T
        if not np.all(np.isfinite(vox)):raise ValueError('nonfinite actual transform')
        fields.append((vox[:,:3],np.asarray(faces)))
    slices=[int(np.clip(np.rint(np.median(fields[-1][0][:,axis])),0,data.shape[axis]-1))for axis in range(3)]
    colors=['#b7b7b7','#39a9e0','#ff7359'];labels=['own nofix','own premesh','own orig']
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),dpi=160)
    segment_counts=[]
    for axis,ax in enumerate(axes):
        remaining=[a for a in range(3)if a!=axis];plane=slices[axis];ax.imshow(np.take(data,plane,axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=180)
        counts=[]
        for (vertices,faces),color,label in zip(fields,colors,labels):
            tri=vertices[faces];dist=tri[:,:,axis]-plane;intersections=[]
            for a,b in ((0,1),(1,2),(2,0)):
                mask=(dist[:,a]<=0)&(dist[:,b]>0)|(dist[:,b]<=0)&(dist[:,a]>0)
                idx=np.flatnonzero(mask);alpha=-dist[idx,a]/(dist[idx,b]-dist[idx,a]);point=tri[idx,a]+alpha[:,None]*(tri[idx,b]-tri[idx,a]);intersections.append((idx,point[:,remaining]))
            grouped={}
            for ids,points in intersections:
                for i,p in zip(ids,points):grouped.setdefault(int(i),[]).append(p)
            segments=np.asarray([p for p in grouped.values()if len(p)==2]);counts.append(len(segments))
            if len(segments):ax.add_collection(LineCollection(segments,colors=color,linewidths=.55,alpha=.85,label=label))
        segment_counts.append(counts);ax.set_title(['sagittal','coronal','axial'][axis]+' / voxel '+str(plane));ax.set_xlim(0,data.shape[remaining[0]]);ax.set_ylim(0,data.shape[remaining[1]]);ax.set_axis_off()
    axes[0].legend(loc='lower left',fontsize=8,facecolor='black',labelcolor='white')
    fig.suptitle('Own topology repair and remesh on the same MRI; contours are not native-reference errors',fontsize=10);fig.tight_layout()
    out.mkdir(mode=0o700,parents=True,exist_ok=False);fig.savefig(out/'brain_contours.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),dpi=150)
    x=np.arange(1,10);values=[r['seconds']['complete_api']for r in prem['per_defect']]
    axes[0].bar(x,values,color='#2985ab');axes[0].set_xlabel('ordered defect (1-based)');axes[0].set_ylabel('complete defect API / seconds');axes[0].set_xticks(x)
    names=['standard remesh + IO','orig cleanup + IO','final QC'];vals=[orig['stages'][k]for k in ('standard_remesh_including_io_seconds','original_cleanup_including_io_seconds','final_structural_quality_seconds')]
    axes[1].barh(names,vals,color='#bb694b');axes[1].set_xlabel('observed stage seconds');fig.suptitle('Four CPU threads; shared-node timing, no whole-recon speed claim',fontsize=10);fig.tight_layout();fig.savefig(out/'observed_stages.png');plt.close(fig)
    receipt=dict(status='complete_actual_all9_orig_figures',source_manifest_sha256=source_manifest_sha256,tool_sha256=sha(__file__),sampler_source_sha256=sha(module.__file__),brain_sha256=sha(brain),surface_sha256=expected,metrics_sha256={n:sha(U/n/'metrics.json')for n in records},slice_voxel_indices=slices,contour_segments=segment_counts,interpretation='own topology/remesh motion, not native-reference error',whole_recon_all='not_assessed',payload={p.name:dict(sha256=sha(p),size=p.stat().st_size)for p in out.glob('*.png')})
    (out/'PLOT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-source','publication-run','premesh-run','orig-run','nofix','brain','output-directory'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-manifest-sha256',required=True);print(json.dumps(plot_completed(**vars(p.parse_args()))))
if __name__=='__main__':main()
