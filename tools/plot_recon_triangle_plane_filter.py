"""真实表面交点标记的正交投影；只导出派生图，不导出坐标。"""
import argparse,hashlib,json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--marks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import nibabel.freesurfer.io as fs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vertices,faces=fs.read_geometry(str(args.input));marks=np.load(args.marks,allow_pickle=False)
    if marks.dtype != np.bool_ or marks.shape != (len(vertices),):raise ValueError('matching (N,) bool vertex marks required')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for axis,dims,label in zip(axes,[(1,2),(0,2),(0,1)],['Sagittal','Coronal','Axial']):
        axis.scatter(vertices[::3,dims[0]],vertices[::3,dims[1]],s=.1,c='#9aabb8',alpha=.25,rasterized=True)
        axis.scatter(vertices[marks,dims[0]],vertices[marks,dims[1]],s=10,c='#ce3737',label='Marked vertices')
        axis.set_aspect('equal');axis.set_title(label);axis.set_xlabel('surface RAS (mm)');axis.set_ylabel('surface RAS (mm)')
    fig.suptitle('Real premesh: identical intersection marks across backends')
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=180);plt.close(fig)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    args.output.with_suffix('.receipt.json').write_text(json.dumps({'input_sha256':digest(args.input),'marks_sha256':digest(args.marks),'figure_sha256':digest(args.output),'plotter_sha256':digest(Path(__file__)),'vertices':len(vertices),'marked_vertices':int(marks.sum()),'visual_scope':'orthogonal projection of real surface; no anatomical image or mesh payload exported'},indent=2)+'\n')
if __name__=='__main__':main()
