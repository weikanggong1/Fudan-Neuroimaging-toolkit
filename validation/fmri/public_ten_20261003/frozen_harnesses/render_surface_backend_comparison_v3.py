"""只读绘制公开 CON01 完整 surface 结果；共享真实32k graymid只用于展示。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_files(path):
    value = json.loads(path.read_text())
    return value.get('result', value)


def cortical_data(path):
    image = nib.load(path)
    if image.shape != (180, 91282):
        raise ValueError('the plot requires the actual complete 180-frame/91k result')
    series, model = image.header.get_axis(0), image.header.get_axis(1)
    if series.unit != 'SECOND' or series.step != 2.1:
        raise ValueError('the actual full-run TR differs from public CON01')
    data = np.asarray(image.dataobj, np.float64)
    if not np.isfinite(data).all():
        raise ValueError('complete saved CIFTI contains nonfinite values')
    arrays = {}
    for structure, selection, subset in model.iter_structures():
        if structure not in ('CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT'):
            continue
        hemisphere = 'left' if structure.endswith('_LEFT') else 'right'
        if subset.nvertices[structure] != 32492 or not subset.surface_mask.all():
            raise ValueError('the actual cortical brain model is not fsLR32k')
        arrays[hemisphere] = (np.asarray(subset.vertex), data[:, selection])
    if set(arrays) != {'left', 'right'}:
        raise ValueError('both actual cortical brain models are required')
    return arrays, series, model


def full_vertex_map(indices, values):
    result = np.full(32492, np.nan, np.float64)
    result[indices] = values
    return result


def difference_maps(reference, candidate):
    maps, metrics = {}, {}
    for hemisphere in ('left', 'right'):
        indices, first = reference[hemisphere]
        candidate_indices, second = candidate[hemisphere]
        if not np.array_equal(indices, candidate_indices) or first.shape != second.shape:
            raise ValueError('actual cortical scalar indices differ')
        rmse = np.sqrt(np.mean((second-first)**2, axis=0))
        scale = np.sqrt(np.mean(first**2, axis=0))
        nrmse = np.full(scale.shape, np.nan)
        np.divide(rmse, scale, out=nrmse, where=scale>0)
        centered_first = first-first.mean(axis=0)
        centered_second = second-second.mean(axis=0)
        denominator = np.sqrt((centered_first**2).sum(axis=0)*(centered_second**2).sum(axis=0))
        correlation = np.full(denominator.shape, np.nan)
        np.divide((centered_first*centered_second).sum(axis=0), denominator,
                  out=correlation, where=denominator>0)
        correlation = np.clip(correlation, -1, 1)
        maps[hemisphere] = {'nrmse':full_vertex_map(indices,nrmse),
                            'temporal_r':full_vertex_map(indices,correlation),
                            'mean':full_vertex_map(indices,first.mean(axis=0))}
        defined = np.isfinite(nrmse)
        defined_r = np.isfinite(correlation)
        metrics[hemisphere] = {'maximum_absolute_difference':float(np.abs(second-first).max()),
            'all_values_exact':bool(np.array_equal(first,second)),
            'nrmse_defined_vertex_count':int(defined.sum()),
            'nrmse_undefined_vertex_count':int((~defined).sum()),
            'nrmse_maximum':float(nrmse[defined].max()) if defined.any() else None,
            'temporal_r_defined_vertex_count':int(defined_r.sum()),
            'temporal_r_minimum':float(correlation[defined_r].min()) if defined_r.any() else None}
    return maps,metrics


def draw_surface(axes, geometry, values, norm, colormap, azimuth):
    vertices, faces = geometry
    values_by_face = values[faces]
    valid = np.isfinite(values_by_face).all(axis=1)
    colors = np.tile([.75,.75,.75,1.],(len(faces),1))
    colors[valid] = colormap(norm(values_by_face[valid].mean(axis=1)))
    collection = Poly3DCollection(vertices[faces],facecolors=colors,edgecolors='none',linewidths=0,rasterized=True)
    axes.add_collection3d(collection)
    axes.auto_scale_xyz(vertices[:,0],vertices[:,1],vertices[:,2])
    axes.set_box_aspect(np.maximum(np.ptp(vertices,axis=0),1))
    axes.view_init(elev=0,azim=azimuth)
    axes.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs-config',type=Path,required=True)
    parser.add_argument('--render-config',type=Path,required=True)
    parser.add_argument('--render-helper',type=Path,required=True)
    parser.add_argument('--data-manifest',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args = parser.parse_args()
    pairs = json.loads(args.pairs_config.read_text())
    render = json.loads(args.render_config.read_text())['render']
    manifest = json.loads(args.data_manifest.read_text())
    public_case = next(value for value in manifest['subjects'] if value['subject']=='CON01')
    if manifest['license']!='CC0' or public_case['complete_original_frames']!=180:
        raise ValueError('only verified public CC0 CON01 full-run data is permitted')
    args.output_root.mkdir(mode=0o700,exist_ok=False)
    paths = {'render_config':args.render_config,'pairs_config':args.pairs_config,'renderer':Path(__file__),
             'mature_mesh_helper':args.render_helper,'public_data_manifest':args.data_manifest}
    dependency=args.render_helper.with_name('compare_subject.py')
    if not dependency.is_file():
        raise FileNotFoundError('the fixed mature renderer dependency is absent')
    paths['mature_mesh_helper_dependency']=dependency
    provenance_entry = render['display_geometry']['provenance']
    paths['geometry_provenance'] = Path(provenance_entry['path'])
    if sha256(paths['geometry_provenance'])!=provenance_entry['sha256']:
        raise ValueError('shared real graymid display provenance changed')
    spec = importlib.util.spec_from_file_location('verified_mature_brain_renderer',args.render_helper)
    helper = importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(args.render_helper.parent))
    spec.loader.exec_module(helper)
    geometries,mesh_identities = {},{}
    for hemisphere in ('left','right'):
        geometries[hemisphere],mesh_identities[hemisphere],path = helper.mesh(render['cortical_meshes'][hemisphere])
        paths['display_geometry/'+hemisphere] = path
    prepared = []
    for pair in pairs['pairs']:
        reports = [Path(pair[key]) for key in ('reference_report','candidate_report')]
        for role,report_path in zip(('reference','candidate'),reports):
            report = json.loads(report_path.read_text())
            if (report['status']!='complete' or report.get('source_revision')!='1128bc52c7a0233266e5b8a8d7dc0b382994e676'
                    or not (report.get('readonly_input_guards_equal')
                            or (report.get('input_unchanged_during_run') and report.get('source_unchanged_during_run')))):
                raise ValueError('only complete guarded backend results can be plotted')
            for key in ('T1w','BOLD'):
                expected = public_case[key]['sha256']
                if 'readonly_inputs_before' in report:
                    matching = [value for name,value in report['readonly_inputs_before'].items()
                                if name.startswith('raw/') and name.endswith('_T1w.nii.gz' if key=='T1w' else '_bold.nii.gz')]
                else:
                    matching = [report['input_sha256']['t1w' if key=='T1w' else 'bold']]
                if matching!=[expected]:
                    raise ValueError('backend raw lineage differs from the public CC0 manifest')
            paths[pair['name']+'/'+role+'_report'] = report_path
            file_map = Path(pair[role+'_files'])
            paths[pair['name']+'/'+role+'_private_file_map'] = file_map
            paths[pair['name']+'/'+role+'_CIFTI'] = Path(selected_files(file_map)['dtseries'])
        prepared.append(pair)
    before = {name:sha256(path) for name,path in paths.items()}
    figure = plt.figure(figsize=(15,4.3*len(prepared)),constrained_layout=True,facecolor='white')
    metrics = {}
    for row,pair in enumerate(prepared):
        reference,series,model = cortical_data(paths[pair['name']+'/reference_CIFTI'])
        candidate,candidate_series,candidate_model = cortical_data(paths[pair['name']+'/candidate_CIFTI'])
        if series!=candidate_series or model!=candidate_model:
            raise ValueError('complete saved CIFTI axes differ')
        maps,metrics[pair['name']] = difference_maps(reference,candidate)
        maximum = max((value['nrmse_maximum'] or 0) for value in metrics[pair['name']].values())
        error_norm = Normalize(0,maximum if maximum>0 else 1)
        norm_r = Normalize(-1,1)
        row_axes=[]
        for column,(hemisphere,kind) in enumerate((('left','nrmse'),('right','nrmse'),('left','temporal_r'),('right','temporal_r'))):
            axes=figure.add_subplot(len(prepared),4,row*4+column+1,projection='3d')
            row_axes.append(axes)
            norm=error_norm if kind=='nrmse' else norm_r
            cmap=plt.get_cmap('viridis' if kind=='nrmse' else 'coolwarm')
            draw_surface(axes,geometries[hemisphere],maps[hemisphere][kind],norm,cmap,180 if hemisphere=='left' else 0)
            axes.set_title(pair['title']+'\n'+('LH' if hemisphere=='left' else 'RH')+' '+kind,fontsize=10)
        error_bar=figure.colorbar(plt.cm.ScalarMappable(norm=error_norm,cmap='viridis'),ax=row_axes[:2],shrink=.6,pad=.01,
            label='NRMSE / reference RMS')
        if maximum==0:
            error_bar.ax.set_title('All defined = 0',fontsize=8)
        figure.colorbar(plt.cm.ScalarMappable(norm=norm_r,cmap='coolwarm'),ax=row_axes[2:],shrink=.6,pad=.01,label='Temporal Pearson r')
    figure.suptitle('Public CC0 CON01: complete saved surface comparisons\nShared source1128 CON01 graymid in fsLR32k; gray = medial wall / undefined; face colors for display only',fontsize=11)
    image=args.output_root/'backend_surface_comparisons.png'
    figure.savefig(image,dpi=170)
    plt.close(figure)
    after={name:sha256(path) for name,path in paths.items()}
    if before!=after:
        raise ValueError('a readonly source, saved result, or shared geometry changed during rendering')
    report={'status':'complete','scope':'Read-only complete180-frame backend saved-output maps; shared real source1128 CON01 graymid display geometry, no MRI or original data export',
        'dataset':'OpenNeuro ds001226 v5.0.1','public_dataset_url':'https://openneuro.org/datasets/ds001226/versions/5.0.1','license':'CC0',
        'raw_sha256':{key:public_case[key]['sha256'] for key in ('T1w','BOLD')},
        'shared_display_geometry':mesh_identities,'display_geometry_provenance_sha256':provenance_entry['sha256'],
        'scalar_index_rule':'actual CIFTI cortical brain-model vertex indices; no fitting or relabeling of reconstruction producers',
        'metrics':metrics,'input_sha256_before':before,'input_sha256_after':after,'input_guards_equal':True,
        'renderer_sha256':before['renderer'],'mature_mesh_helper_sha256':before['mature_mesh_helper'],
        'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,'nibabel_version':nib.__version__,
        'figure_sha256':sha256(image)}
    (args.output_root/'figure.public.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('BACKEND_BRAIN_FIGURE_SAVED',report['figure_sha256'])


if __name__=='__main__':
    main()
