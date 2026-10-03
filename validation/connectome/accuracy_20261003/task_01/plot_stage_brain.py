"""Native-grid brain slices for the matched real-data EDDY stage comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("reference","baseline","candidate","mask","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--frame",type=int,default=0)
    p.add_argument("--slice",type=int,default=30)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    paths=[args.reference,args.baseline,args.candidate]
    images=[nib.load(str(path)) for path in paths]
    if any(x.shape!=images[0].shape or not np.allclose(x.affine,images[0].affine,atol=1e-5,rtol=0)
           for x in images):raise ValueError("all images must use the same original grid")
    data=[np.asarray(x.dataobj[...,args.frame],dtype=np.float32) for x in images]
    mask=np.asarray(nib.load(str(args.mask)).dataobj)>0
    vmax=float(np.percentile(data[0][mask],99))
    errors=[np.where(mask, x-data[0], 0) for x in data[1:]]
    emax=max(float(np.percentile(np.abs(x[mask]),99)) for x in errors)
    emax=max(emax,1e-6)
    arrays=data+errors
    labels=["Official CPU","Frozen FNIT","Candidate FNIT","Frozen - official","Candidate - official"]
    canvas=Image.new("RGB",(5*260,310),"white");draw=ImageDraw.Draw(canvas)
    for i,(array,label) in enumerate(zip(arrays,labels)):
        plane=np.rot90(array[:,:,args.slice])
        if i<3:
            value=np.clip(plane/vmax,0,1)
            rgb=np.repeat((255*value).astype(np.uint8)[...,None],3,axis=-1)
        else:
            value=np.clip(plane/emax,-1,1)
            rgb=np.stack([255*(1-np.minimum(-value,1).clip(0,1)),
                          255*(1-np.abs(value)),255*(1-np.minimum(value,1).clip(0,1))],axis=-1).astype(np.uint8)
        panel=Image.fromarray(rgb).resize((250,250),Image.Resampling.NEAREST)
        canvas.paste(panel,(i*260+5,30));draw.text((i*260+5,5),label,fill="black")
    draw.text((5,286),f"Native frame={args.frame}; z={args.slice}; intensity=[0,{vmax:.3g}]; error=[-{emax:.3g},{emax:.3g}]",fill="black")
    args.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(args.output)
    receipt={"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "input_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths+[args.mask]},
             "frame":args.frame,"slice":args.slice,"intensity_limits":[0,vmax],"difference_limits":[-emax,emax],
             "resampling":"none; nearest display zoom only", "difference_display":"official brain mask only",
             "dataset_license":"CC0, ds001226",
             "shape":list(images[0].shape),"affine":images[0].affine.tolist()}
    args.output.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":main()
