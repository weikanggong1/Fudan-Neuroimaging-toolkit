"""已完成真实 CON03 FA、FS-aparc与固定TCK端点的脑图；仅显示不修改输入。"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from benchmark_assignment_weights import checked,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise ValueError('new figure path required')
    common=json.loads(args.reference.read_text());producer_path=checked(common['producer']['path'],common['producer']['sha256'])
    producer=json.loads(producer_path.read_text());atlas_info=common['profiles']['fs-aparc']['atlas']
    atlas_image=nib.load(checked(atlas_info['path'],atlas_info['sha256']));atlas=np.asarray(atlas_image.dataobj).astype(np.int32)
    fa_info=producer['source']['images']['fa'];fa_image=nib.load(checked(fa_info['path'],fa_info['sha256']));fa=fa_image.get_fdata(dtype=np.float32)
    if fa.shape!=atlas.shape or not np.allclose(fa_image.affine,atlas_image.affine,rtol=0,atol=1e-6):raise ValueError('same FA/atlas grid required')
    tracks_info=common['inputs']['tracks.tck'];paths=nib.streamlines.load(str(checked(tracks_info['path'],tracks_info['sha256']))).tractogram.streamlines
    endpoints=np.stack([(v[0],v[-1]) for v in paths]).reshape(-1,3).astype(np.float64)
    ijk=nib.affines.apply_affine(np.linalg.inv(atlas_image.affine),endpoints)
    indices=np.floor(ijk+.5).astype(np.int64);inside=((indices>=0)&(indices<atlas.shape)).all(-1)
    z=int(np.bincount(indices[inside,2],minlength=atlas.shape[2]).argmax())
    # Display finite FA as grey; nonfinite measurements remain marked purple.
    fa_slice=fa[:,:,z].T[::-1];grey=(np.clip(np.where(np.isfinite(fa_slice),fa_slice,0),0,1)*255).astype(np.uint8)
    base=np.repeat(grey[...,None],3,axis=-1);base[~np.isfinite(fa_slice)]=[180,0,180]
    labels=atlas[:,:,z].T[::-1];colors=np.stack([(labels*53)%256,(labels*97)%256,(labels*193)%256],-1).astype(np.uint8)
    colors[labels==0]=base[labels==0]
    projected=base.copy();selected=indices[inside&(indices[:,2]==z)]
    for x,y,_ in selected:projected[atlas.shape[1]-1-y,x]=[255,170,25]
    canvas=Image.new('RGB',(3*384,440),'white');draw=ImageDraw.Draw(canvas)
    titles=['Official FA','FS-aparc in DWI world','Fixed TCK endpoints']
    for i,array in enumerate([base,colors,projected]):
        im=Image.fromarray(array).resize((384,384),Image.Resampling.NEAREST);canvas.paste(im,(384*i,30));draw.text((384*i+10,8),titles[i],fill='black')
    draw.text((12,419),f'OpenNeuro ds001226 CON03 | native z={z} | {len(paths)} official streamlines | visualization only',fill='black')
    canvas.save(args.output)
    sidecar={'scope':'real public brain/atlas/endpoints visualization; no anatomical resampling; not accuracy metric',
        'harness_sha256':sha(__file__),'reference_sha256':sha(args.reference),'producer':common['producer'],
        'atlas':atlas_info,'fa':{'path':fa_info['path'],'sha256':fa_info['sha256']},'tracks':tracks_info,
        'slice_z':z,'affine':atlas_image.affine.tolist(),'display_policy':'finite FA clipped0..1 only for grey display; nonfinite purple; label colors deterministic; projected endpoints floor(ijk+.5) only for display, not production assignment',
        'output_sha256':sha(args.output)}
    args.output.with_suffix('.json').write_text(json.dumps(sidecar,indent=2)+'\n')


if __name__=='__main__':main()
