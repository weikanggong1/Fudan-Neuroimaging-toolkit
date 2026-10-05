"""Plot the unchanged saved CC0 target masks; no preparation, interpolation or registration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run-directory",type=Path,required=True);a=p.parse_args()
    os.umask(0o077)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    root=a.run_directory;result=json.loads((root/"scalar_metadata_recovery_once/result.private.json").read_text())
    assert result["exit"]==0 and result["measurements"]["all_gates_pass"]
    inputs=[root/arm/"targetMask.mgz" for arm in ("official","fnit")]
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    for arm,q in zip(("official","fnit"),inputs):assert sha(q)==result["saved_outputs_after"][arm+"/targetMask.mgz"]["sha256"]
    images=[nib.load(q) for q in inputs];original=[np.asarray(image.dataobj)>0 for image in images]
    assert np.array_equal(original[0],original[1]) and np.array_equal(images[0].affine,images[1].affine)
    # Exact permutation/reflection to RAS for display, without interpolation.
    orient=nib.orientations.io_orientation(images[0].affine)
    change=nib.orientations.ornt_transform(orient,nib.orientations.axcodes2ornt(("R","A","S")))
    masks=[nib.orientations.apply_orientation(mask,change) for mask in original]
    indices=np.argwhere(masks[0]|masks[1]);centers=((indices.min(0)+indices.max(0))//2).tolist()
    names=("Sagittal","Coronal","Axial");figure,axes=plt.subplots(2,3,figsize=(8,5.2),facecolor="white")
    for row,(arm,mask) in enumerate(zip(("Installed SAMSEG prep","FNIT PyTorch prep"),masks)):
        for axis in range(3):
            panel=axes[row,axis];panel.imshow(np.take(mask,centers[axis],axis=axis).T,origin="lower",cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            panel.set_xticks([]);panel.set_yticks([])
            if row==0:panel.set_title(names[axis],fontsize=11)
            if axis==0:panel.set_ylabel(arm,fontsize=10)
    figure.suptitle("Right HA alignment target: 0 differing voxels",fontsize=13)
    figure.text(.5,.02,"Same saved grid; source ASEG 1 mm (resize not triggered). Public CC0-derived mask only.",ha="center",fontsize=8)
    figure.subplots_adjust(left=.11,right=.99,bottom=.07,top=.87,wspace=.04,hspace=.10)
    output=root/"scalar_metadata_recovery_once/preparation_targets.png";assert not output.exists();figure.savefig(output,dpi=135);plt.close(figure)
    metadata={"scope":"derived_CC0_binary_target_plot_only_no_interpolation_or_registration","source_sha256":sha(Path(__file__)),
      "figure_sha256":sha(output),"target_input_sha256":{arm:sha(q) for arm,q in zip(("official","fnit"),inputs)},
      "source_grid_shape":[int(v) for v in original[0].shape],"display_RAS_shape":[int(v) for v in masks[0].shape],
      "display_union_bbox_midpoint_indices":[int(v) for v in centers],"saved_images_unchanged":True,"resize_triggered":False}
    output.with_suffix(".private.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"png_bytes":output.stat().st_size,"png_sha256":sha(output)}))


if __name__=="__main__":main()
