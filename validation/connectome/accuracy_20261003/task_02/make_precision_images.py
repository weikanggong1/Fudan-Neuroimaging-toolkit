"""公开 CON10 实际 corrected 输入 FA 精度脑图；统一误差色标。"""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path,required=True)
    parser.add_argument('--diagnostic-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('fresh image path required')
    case='CON10';refs=json.loads((args.reference_root/'report.json').read_text())
    mask_path=next(p for p in refs['cases'][case]['sources'] if p.endswith('brain_mask_dwi.nii.gz'))
    mask=np.asarray(nib.load(mask_path).dataobj)>0
    official=nib.load(args.reference_root/case/'native_fslgrad_fa.nii.gz').get_fdata(dtype=np.float32)
    baseline=np.load(args.diagnostic_root/(case+'_baseline_response_baseline_grad_vs_native_fslgrad.npz'))['fa']
    candidate=np.load(args.diagnostic_root/(case+'_candidate_response_candidate_grad_vs_native_fslgrad.npz'))['fa']
    old_error=np.where(mask,np.abs(baseline-official),np.nan)
    new_error=np.where(mask,np.abs(candidate-official),np.nan)
    worst=np.unravel_index(np.nanargmax(old_error),old_error.shape)
    slices=[30,int(worst[2])]
    fig=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(fig)
    font=ImageFont.load_default(size=18);small=ImageFont.load_default(size=16)
    draw.text((20,15),'Public ds001226 CON10: identical formal corrected DWI / mask',fill='black',font=font)
    draw.text((20,45),f'Full-mask maximum FA error: {np.nanmax(old_error):.6g} -> {np.nanmax(new_error):.6g}',fill='black',font=font)
    draw.text((20,75),f'Full-mask voxels >1e-5: {np.nansum(old_error>1e-5)} -> {np.nansum(new_error>1e-5)}',fill='black',font=font)
    for row,z in enumerate(slices):
        arrays=[np.where(mask,official,np.nan),old_error,new_error]
        for col,array in enumerate(arrays):
            values=array[:,:,z].T[::-1]
            scale=np.nan_to_num(values,nan=0,posinf=0,neginf=0)/(1 if col==0 else 1e-4)
            scale=np.clip(scale,0,1)
            if col==0:rgb=np.repeat((scale[...,None]*255).astype(np.uint8),3,-1)
            else:
                # Shared black-red-yellow map; both error panels use [0,1e-4].
                rgb=np.stack((np.clip(scale*2,0,1),np.clip(scale*2-1,0,1),np.zeros_like(scale)),-1)
                rgb=(rgb*255).astype(np.uint8)
            rgb[np.isnan(values)]=255
            panel=Image.fromarray(rgb).resize((330,330),Image.Resampling.NEAREST)
            x,y=20+col*400,150+row*375;fig.paste(panel,(x,y))
            draw.text((x,y-25),['Native FA [0,1]','Original FNIT error','Double import + unit norm'][col],fill='black',font=small)
            draw.text((x,y+335),f'voxel x: 0..95; y: 95..0; z={z}',fill='gray',font=small)
            if row==1:
                point=(x+worst[0]*330/96,y+(95-worst[1])*330/96)
                draw.ellipse((point[0]-4,point[1]-4,point[0]+4,point[1]+4),outline='cyan',width=2)
    draw.text((20,875),'Error panels share linear [0,1e-4] scale; black=0, yellow>=1e-4; cyan marks worst original voxel.',fill='black',font=small)
    fig.save(args.output)


if __name__=='__main__':main()
