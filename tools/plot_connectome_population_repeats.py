"""绘制已保存的真实五种子人口分布与 point-visit 脑图；不参与数值门槛。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(report_path, data_path, provenance_path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report = json.loads(report_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if digest(report_path) != provenance['population_report_sha256'] or digest(data_path) != provenance['plot_data_sha256']:
        raise ValueError('plot inputs changed from actual source-verified analysis')
    data = np.load(data_path, allow_pickle=False)
    official, fnit = data['official_mean_tdi'], data['fnit_mean_tdi']
    if official.shape != fnit.shape or list(official.shape) != report['grid_shape']:
        raise ValueError('plot grid differs from actual population metric grid')
    output.mkdir(parents=True, exist_ok=True)
    planes = []
    for axis in range(3):
        center = official.shape[axis] // 2
        planes.append((np.take(official, range(center - 2, center + 3), axis=axis).sum(axis=axis).T,
                       np.take(fnit, range(center - 2, center + 3), axis=axis).sum(axis=axis).T))
    vmax = float(np.quantile(np.concatenate([m.ravel() for plane in planes for m in plane]), .995))
    delta = max(float(np.quantile(np.concatenate([(b-a).ravel() for a,b in planes]), .005)),
                -float(np.quantile(np.concatenate([(b-a).ravel() for a,b in planes]), .995)), key=abs)
    delta = abs(delta)
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    titles = ['center x: y/z voxels', 'center y: x/z voxels', 'center z: x/y voxels']
    for col, ((left, right), title) in enumerate(zip(planes, titles)):
        positive_image = axes[0,col].imshow(left, origin='lower', cmap='magma', vmin=0, vmax=vmax)
        axes[1,col].imshow(right, origin='lower', cmap='magma', vmin=0, vmax=vmax)
        difference_image = axes[2,col].imshow(right-left, origin='lower', cmap='coolwarm', vmin=-delta, vmax=delta)
        axes[0,col].set_title(title)
        for ax in axes[:,col]: ax.set_xlabel('native FOD voxel index'); ax.set_ylabel('native FOD voxel index')
    for ax,label in zip(axes[:,0], ['MRtrix: mean of 5 runs', 'FNIT: mean of 5 runs', 'FNIT minus MRtrix']):
        ax.annotate(label, xy=(-.30,.5), xycoords='axes fraction', rotation=90, ha='center', va='center')
    fig.colorbar(positive_image, ax=axes[:2,:].ravel().tolist(), shrink=.7, label='Stored point visits: mean across 5 runs / sum of 5 slices')
    fig.colorbar(difference_image, ax=axes[2,:].ravel().tolist(), shrink=.7, label='FNIT minus MRtrix: same stored-point visit units')
    fig.suptitle('ds001226 sub-CON03 preop: actual stored-point visits\n5 central voxel slices per plane; same native FOD grid; not MRtrix tckmap', fontsize=13)
    fig.savefig(output/'point_visit_brain.png', dpi=170);plt.close(fig)
    fields=[('accepted_fraction_absolute_difference','Accepted fraction absolute difference'),('length_ks','Length KS'),
            ('endpoint_8mm_histogram_pearson','Endpoint 8 mm histogram Pearson'),
            ('tdi_native_voxel_pearson','Native stored-point visit Pearson'),
            ('tdi_four_voxel_block_pearson','4-voxel-block stored-point visit Pearson')]
    fig, axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,(field,title) in zip(axes.ravel(),fields):
        row=report['ranges'][field]
        values=np.asarray(row['fnit_vs_official'],dtype=float).reshape(len(report['official_seeds']),len(report['fnit_seeds']))
        accepted=np.asarray(row['comparison_accepted'],dtype=object).reshape(values.shape)
        im=ax.imshow(values,cmap='viridis',aspect='equal');fig.colorbar(im,ax=ax,shrink=.75)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j,i,f'{values[i,j]:.4f}',ha='center',va='center',color='white',fontsize=8)
                if accepted[i,j] is False: ax.add_patch(plt.Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor='#ff3333',linewidth=2))
        ax.set(xlabel='FNIT seed',ylabel='MRtrix seed',xticks=range(values.shape[1]),yticks=range(values.shape[0]),
               title=f'{title}\n{row["criterion"]}: {row["threshold"]:.7g}; {row["accepted_count"]}/{values.size} pass')
    ax=axes.ravel()[-1]
    bins=np.arange(0,255,5)
    for prefix,color in [('official','#2374ab'),('fnit','#d55e00')]:
        for i in range(len(report[prefix+'_seeds'])):
            ax.hist(data[f'{prefix}_{i}_lengths'], bins=bins, density=True, histtype='step',color=color,alpha=.7,
                    label=prefix if i==0 else None)
    ax.legend(frameon=False);ax.set(xlabel='Stored polyline length (mm)',ylabel='Density',title='Actual accepted streamline populations')
    fig.suptitle('CON03: all 25 MRtrix/FNIT comparisons; red frames fail predeclared one-sided gate\nEndpoint uses 8 mm bins shared by all 10 runs; finite observed range, not a confidence interval',fontsize=12)
    fig.savefig(output/'population_cross_pairs.png',dpi=170);plt.close(fig)
    return {'plot_script_sha256':digest(Path(__file__)), 'report_sha256':digest(report_path),
        'plot_data_sha256':digest(data_path), 'images':{p.name:digest(p) for p in output.glob('*.png')},
        'scope':'display only; original metrics/gates and all recorded source data unchanged'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--plot-data',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    result=plot(args.report,args.plot_data,args.provenance,args.output_dir)
    (args.output_dir/'plot_provenance.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
