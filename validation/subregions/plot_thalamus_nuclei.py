"""Six axial fine-nucleus overlays from saved real T1 labels; CPU only."""
from pathlib import Path
import argparse,hashlib,json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
from nibabel.processing import resample_from_to,resample_to_output
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
for name in ("t1","reference","candidate","lut","output"):
 parser.add_argument("--"+name,type=Path,required=True)
parser.add_argument("--input-scope",choices=("stage","raw"),default="stage",
                    help="Describe supplied official stage T1 or raw T1 in the figure and metadata.")
args=parser.parse_args()
def identity(path):
 data=path.read_bytes()
 return {"path":str(path),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
original_image=nib.load(args.t1)
canonical_image=nib.as_closest_canonical(original_image)
canonical_spacing=np.linalg.norm(canonical_image.affine[:3,:3],axis=0)
linear=canonical_image.affine[:3,:3]
off_diagonal=linear-np.diag(np.diag(linear))
alignment_tolerance=1e-5*max(1.,float(canonical_spacing.max()))
align_axes=bool(np.max(np.abs(off_diagonal))>alignment_tolerance)
# Canonical reorientation preserves acquisition obliquity. Only genuinely
# oblique inputs need another sampling step; near-identity stage affines retain
# their exact grid instead of acquiring an extra boundary voxel from roundoff.
image=(resample_to_output(canonical_image,voxel_sizes=canonical_spacing,order=1)
       if align_axes else canonical_image)
grid=(image.shape,image.affine)
spacing=np.linalg.norm(image.affine[:3,:3],axis=0)
pixel_aspect=float(spacing[1]/spacing[0])
data=np.asarray(image.dataobj,np.float32)
volumes=[np.asarray(resample_from_to(nib.load(path),grid,order=0).dataobj,np.int32)
         for path in (args.reference,args.candidate)]
volumes=[np.where((v>=8100)&(v<8300)&~np.isin(v,(8125,8225)),v,0) for v in volumes]
points=np.argwhere(volumes[0]!=0)
all_points=np.argwhere((volumes[0]!=0)|(volumes[1]!=0))
low=np.maximum(all_points.min(0)-8,0)
high=np.minimum(all_points.max(0)+9,image.shape)
crop=(slice(low[0],high[0]),slice(low[1],high[1]))
zmin,zmax=points[:,2].min(),points[:,2].max()
indices=np.round(zmin+(np.arange(6)+.5)/6*(zmax-zmin)).astype(int)
lookup={}
for line in args.lut.read_text().splitlines():
 fields=line.split()
 if len(fields)<7:continue
 label=int(fields[0])
 if 8100<=label<8300 and label not in (8125,8225):
  lookup[label]={"name":fields[2],"rgb":[int(v) for v in fields[3:6]]}
labels=sorted(int(v) for v in np.union1d(np.unique(volumes[0]),np.unique(volumes[1])) if v)
missing=set(labels)-set(lookup)
if missing:raise ValueError("LUT missing observed fine nucleus IDs: "+str(missing))
def overlay(labels):
 rgba=np.zeros((*labels.shape,4),np.float32)
 for label in np.unique(labels):
  if label==0:continue
  rgba[labels==label,:3]=np.asarray(lookup[int(label)]["rgb"])/255
  rgba[labels==label,3]=.86
 return rgba.transpose(1,0,2)
fig,axes=plt.subplots(3,6,figsize=(16,8.4),facecolor="black")
vmax=float(np.percentile(data[data>0],99))
zras=[]
for column,index in enumerate(indices):
 background=data[crop+(index,)].T
 for row,volume in enumerate(volumes):
  ax=axes[row,column]
  ax.imshow(background,cmap="gray",vmin=0,vmax=vmax,origin="lower",interpolation="nearest")
  ax.imshow(overlay(volume[crop+(index,)]),origin="lower",interpolation="nearest")
 ax=axes[2,column]
 ax.imshow(background,cmap="gray",vmin=0,vmax=vmax,origin="lower",interpolation="nearest")
 changed=volumes[0][crop+(index,)]!=volumes[1][crop+(index,)]
 rgba=np.zeros((*changed.shape,4),np.float32)
 rgba[changed]=(1.,.18,.18,.88)
 ax.imshow(rgba.transpose(1,0,2),origin="lower",interpolation="nearest")
 z=float((image.affine@np.array([0,0,index,1]))[2]);zras.append(z)
 axes[0,column].set_title(f"z = {z:.1f} mm",color="white",fontsize=11)
 for row in range(3):
  axes[row,column].text(.015,.015,"L",color="white",transform=axes[row,column].transAxes,fontsize=8)
  axes[row,column].text(.96,.015,"R",color="white",transform=axes[row,column].transAxes,fontsize=8)
for row,title in enumerate(("FreeSurfer 8.2 reference","FNIT","Different nucleus labels")):
 axes[row,0].set_ylabel(title,color="white",fontsize=11)
for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([]);ax.set_aspect(pixel_aspect)
legend={}
for label in labels:
 name=lookup[label]["name"].removeprefix("Left-").removeprefix("Right-")
 legend[name]=lookup[label]["rgb"]
fig.legend(handles=[Patch(color=np.asarray(rgb)/255,label=name) for name,rgb in sorted(legend.items())],
 loc="lower center",ncol=8,fontsize=7,labelcolor="white",facecolor="black",edgecolor="none")
spacing_text=" x ".join(f"{value:.3g}" for value in spacing)
fig.suptitle(f"Real sub-01 {args.input_scope} T1 | display grid {spacing_text} mm | thalamic nuclei",color="white",fontsize=12)
fig.subplots_adjust(left=.045,right=.995,bottom=.16,top=.92,wspace=.025,hspace=.04)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=170,facecolor="black")
plt.close(fig)
metadata={"kind":f"saved real {args.input_scope} T1 CPU-only thalamic fine-nucleus six axial slices",
 "inputs":{name:identity(path) for name,path in (("t1",args.t1),("official_reference",args.reference),("candidate",args.candidate),("color_lut",args.lut))},
 "plot_driver":identity(Path(__file__)),"output":identity(args.output),
 "original_input_grid":{"shape":[int(v) for v in original_image.shape],"affine_RAS":original_image.affine.tolist(),
                        "axis_codes":list(nib.aff2axcodes(original_image.affine))},
 "canonical_input_grid":{"shape":[int(v) for v in canonical_image.shape],"affine_RAS":canonical_image.affine.tolist(),
                         "spacing_mm":canonical_spacing.tolist()},
 "display_pixel_aspect_y_over_x":pixel_aspect,
 "display_grid":{"shape":[int(v) for v in image.shape],"affine_RAS":image.affine.tolist(),"spacing_mm":spacing.tolist()},
 "axis_alignment":{"applied":align_axes,"canonical_off_diagonal_max_mm":float(np.max(np.abs(off_diagonal))),
                   "off_diagonal_tolerance_mm":alignment_tolerance,
                   "method":"resample_to_output to RAS axes" if align_axes else "canonical grid retained; already aligned within tolerance"},
 "t1_display_sampling":{"interpolation":"linear" if align_axes else "none; canonical axis reordering only",
                        "resampling_order":1 if align_axes else None},
 "label_display_sampling":{"interpolation":"nearest","resampling_order":0},
 "slices_RAS_z_mm":zras,"slices_display_indices":[int(v) for v in indices],
 "slice_selection":"six equal-bin centers within saved official thalamus z extent; both maps nearest-neighbour resampled to the same RAS-axis-aligned display T1 grid",
 "legend":{"source":"compressionLookupTable.txt atlas RGB; matching left and right nucleus names share a color","labels":lookup},
 "difference":"red pixels mark any differing fine nucleus label, including background; no thresholding or morphology in the plot",
 "data_scope":f"public defaced OpenNeuro ds000114 sub-01, {args.input_scope} T1; original reference is saved FreeSurfer 8.2 ThalamicNuclei.FSvoxelSpace.mgz"}
args.output.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
print(json.dumps({"output":str(args.output),"bytes":args.output.stat().st_size,"slices_RAS_z_mm":zras,"nuclei_colors":len(legend)}))
