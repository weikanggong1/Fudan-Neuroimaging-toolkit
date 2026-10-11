"""已完成显式缓存阶段的实际墙钟、六评分和自产缺陷MRI叠加。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*,frozen_source,input_run,brain,metrics,output):
    """五具名Path→完成门及输入/状态SHA校验后的新PNG；不运行GA。

    frozen_source为本次实际源码，input_run为已完成pair父目录，brain是
    本阶段同链自身uint8 MRI；metrics为安全白名单，output须不存在。
    两个产物先核完整SHA，再画真实六个分数/冷CLI/完整search API和
    自身最终selected域的同MRI叠加。脑图单位体素，坐标由原MRI头生成
    变换；不是原版误差、整例QC或默认完整495评分性能图。
    """
    root,output,source,metrics=map(Path,(input_run,output,frozen_source,metrics))
    value=json.loads(metrics.read_text())
    exported=json.loads(metrics.with_name('EXPORT_RECEIPT.json').read_text())
    manifest_path=source/'SOURCE_MANIFEST.private.json'
    manifest=json.loads(manifest_path.read_text())
    if (sha(manifest_path)!=value['source_manifest_sha256']
            or exported['source_manifest_sha256']!=value['source_manifest_sha256']
            or exported['payload']['metrics']['sha256']!=sha(metrics)
            or exported['payload']['metrics']['size']!=metrics.stat().st_size
            or not exported['numeric_values_and_types_preserved']):
        raise ValueError('same completed export and frozen source SHA required')
    for name,record in manifest['payload'].items():
        path=source/name
        if path.stat().st_size!=record['size']or sha(path)!=record['sha256']:
            raise ValueError('actual frozen source payload changed')
    if output.exists()or output.suffix.lower()!='.png':raise ValueError('fresh PNG required')
    if (value['status']!='passed_six_score_explicit_face_cache_pair'or value['execution']['exit_code']!=0
            or not all(value['controls'].values())or
            not all(row['all_fields_byte_equal']for row in value['comparisons'].values())):
        raise ValueError('complete strict same-input stage required')
    for row in value['records']:
        if sha(brain)!=row['input_sha256']['brain']:raise ValueError('same own MRI SHA required')
        for name in ('all_scores.private.npz','final_candidate.private.npz'):
            if sha(root/'pair'/row['label']/name)!=row['checkpoint_sha256'][name]:
                raise ValueError('same completed numeric artifact required')
    sys.path.insert(0,str(source/'src'))
    import fnit.recon_all
    fnit.recon_all.__path__.insert(0,str(source/'src/fnit/recon_all'))
    import numpy as np
    import nibabel as nib
    from fnit.recon_all.topology_edge_mri_score import source_topology_ras_to_voxel
    actual_module=sys.modules[source_topology_ras_to_voxel.__module__]
    module_path=Path(actual_module.__file__).resolve()
    expected_path=(source/'src/fnit/recon_all/topology_edge_mri_score.py').resolve()
    if (module_path!=expected_path or sha(module_path)!=manifest['expected_module_sha256'][
            'fnit.recon_all.topology_edge_mri_score']):
        raise ValueError('actual loaded MRI transform source differs from completed stage')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    transform=source_topology_ras_to_voxel(volume_header=image.header)
    values=[];scores=[]
    for row in value['records']:
        folder=root/'pair'/row['label']
        with np.load(folder/'final_candidate.private.npz',allow_pickle=False)as z:
            selected=z['selected_vertices'].copy();xyz=z['original_vertices'][selected].copy()
        voxel=np.c_[xyz,np.ones(len(xyz))]@transform.T
        values.append(voxel[:,:3])
        with np.load(folder/'all_scores.private.npz',allow_pickle=False)as z:scores.append(z['fitness'].copy())
    if scores[0].tobytes()!=scores[1].tobytes():raise ValueError('six score bytes differ')
    center=np.median(values[1],axis=0);plane=int(np.clip(np.rint(center[0]),0,volume.shape[0]-1))
    low=np.maximum(np.floor(np.min(values[1][:,1:],axis=0))-12,0).astype(int)
    high=np.minimum(np.ceil(np.max(values[1][:,1:],axis=0))+13,np.array(volume.shape[1:])).astype(int)
    figure,axes=plt.subplots(1,4,figsize=(16,4.2))
    x=np.arange(2);width=.35
    axes[0].bar(x-width/2,[row['fresh_cli_seconds']for row in value['records']],width,label='cold CLI')
    axes[0].bar(x+width/2,[row['stages']['search_and_final_API']for row in value['records']],width,label='search + final API')
    axes[0].set(xticks=x,xticklabels=['disabled','enabled'],ylabel='seconds',title='Actual six-score stage')
    axes[0].legend(fontsize=7)
    axes[1].plot(np.arange(1,7),scores[0],'-o',label='disabled',markersize=4)
    axes[1].plot(np.arange(1,7),scores[1],'x',label='enabled',markersize=7)
    axes[1].set(xlabel='five initial + final score',ylabel='fitness',title='All six double scores byte equal')
    axes[1].legend(fontsize=7)
    for axis,points,label in zip(axes[2:],values,('disabled','enabled')):
        axis.imshow(volume[plane,low[0]:high[0],low[1]:high[1]].T,cmap='gray',origin='lower',
            extent=[low[0],high[0],low[1],high[1]])
        near=np.abs(points[:,0]-plane)<=2
        axis.scatter(points[near,1],points[near,2],s=8,c='#ff6633',alpha=.8)
        axis.set(xlabel='voxel Y',ylabel='voxel Z',title=f'{label}: own final patch on own MRI')
    figure.suptitle('Same 4-thread budget; paired stage only, no full-GA or whole-pipeline speed claim',fontsize=10)
    figure.tight_layout();output.parent.mkdir(parents=True,exist_ok=True);figure.savefig(output,dpi=160);plt.close(figure)
    return dict(status='complete_plot_of_verified_same_input_stage',scores=6,
        input_sha256=dict(metrics=sha(metrics),brain=sha(brain)),output_sha256=sha(output),
        plotter_sha256=sha(__file__),mri_slice_voxel_x=plane,selected_vertices=len(values[1]),
        source_manifest_sha256=sha(manifest_path),
        actual_loaded_mri_transform_sha256=sha(module_path),
        export_receipt_sha256=sha(metrics.with_name('EXPORT_RECEIPT.json')),
        final_plot_is_own_produced_coordinates=True,original_software_error='not_assessed',
        complete_default_GA=False,whole_recon_all='not_assessed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-source','input-run','brain','metrics','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':
    main()
