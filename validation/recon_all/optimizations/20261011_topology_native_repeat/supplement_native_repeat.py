"""只读原生重复产物：分离坐标、面、数值几何和文件来源字段，并绘制公开脑图。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import nibabel as nib


def sha(data):return hashlib.sha256(data).hexdigest()


def inspect_bytes(path):
    data=path.read_bytes();cursor=3
    if data[:3]!=bytes((255,255,254)):raise ValueError('triangle surface required')
    for _ in range(2):cursor=data.index(b'\n',cursor)+1
    n,f=struct.unpack('>ii',data[cursor:cursor+8]);start=cursor+8;stop=start+n*12+f*12
    _,_,geometry=nib.freesurfer.read_geometry(path,read_metadata=True)
    name=geometry.pop('filename')
    numeric={key:dict(shape=list(np.asarray(value).shape),dtype=str(np.asarray(value).dtype),
        sha256=sha(np.asarray(value).tobytes()))for key,value in geometry.items()}
    return data,dict(bytes=len(data),file_sha256=sha(data),preamble_sha256=sha(data[:start]),
        coordinate_and_face_block_sha256=sha(data[start:stop]),data_start=start,data_end=stop,
        trailer_sha256=sha(data[stop:]),geometry_fields=numeric,
        filename=dict(utf8_sha256=sha(name.encode()),characters=len(name),contains_path_separator='/'in name or '\\'in name))


def run(*,reference,default_repeat,verbose_repeat,brain,output):
    output.mkdir(parents=True,exist_ok=True)
    old,old_fields=inspect_bytes(reference)
    result=dict(scope='read-only post-run surface metadata; no original program execution',
        script_sha256=sha(Path(__file__).read_bytes()),reference=old_fields,comparisons={})
    for label,path in (('default_repeat',default_repeat),('verbose_repeat',verbose_repeat)):
        data,fields=inspect_bytes(path)
        common=min(len(old),len(data));different=[i for i in range(common)if old[i]!=data[i]]
        result['comparisons'][label]=dict(candidate=fields,
            preamble_bytes_exact=old_fields['preamble_sha256']==fields['preamble_sha256'],
            coordinate_and_face_block_bytes_exact=old_fields['coordinate_and_face_block_sha256']==fields['coordinate_and_face_block_sha256'],
            geometry_numeric_bytes_exact=old_fields['geometry_fields']==fields['geometry_fields'],
            header_filename_exact=old_fields['filename']['utf8_sha256']==fields['filename']['utf8_sha256'],
            raw_file_bytes_exact=old==data,first_raw_difference_offset=different[0]if different else common if len(old)!=len(data)else None,
            differing_bytes_in_common_length=len(different),file_length_difference=len(data)-len(old),
            difference_in_trailer=bool(different and different[0]>=old_fields['data_end']))
    result['brain_sha256']=sha(brain.read_bytes())
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image=nib.load(brain);volume=np.asarray(image.dataobj)
    reference_vertices,faces=nib.freesurfer.read_geometry(reference)
    candidate_vertices,_=nib.freesurfer.read_geometry(verbose_repeat)
    to_voxel=np.linalg.inv(image.header.get_vox2ras_tkr())
    points=nib.affines.apply_affine(to_voxel,reference_vertices)
    centre=np.asarray(volume.shape)//2
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    for ax,axis,title in zip(axes,(0,1,2),('Sagittal','Coronal','Axial')):
        remaining=[i for i in range(3)if i!=axis];mask=np.abs(points[:,axis]-centre[axis])<=1.5
        ax.imshow(np.take(volume,centre[axis],axis=axis).T,cmap='gray',origin='lower',vmin=0,vmax=130)
        ax.scatter(points[mask,remaining[0]],points[mask,remaining[1]],s=2,c='#65e6c3',label='native reference = repeat')
        ax.set_title(title);ax.axis('off')
    fig.suptitle('ds000114 sub-07 LH: native GA repeat, coordinate difference = 0 mm')
    figure=output/'native_repeat.png';fig.savefig(figure,dpi=160);plt.close(fig)
    result['figure']=dict(sha256=sha(figure.read_bytes()),bytes=figure.stat().st_size,
        scope='public T1 derivative and native surface overlay; no whole recon-all quality conclusion',
        reference_surface_sha256=sha(reference.read_bytes()),repeat_surface_sha256=sha(verbose_repeat.read_bytes()),
        full_coordinates_byte_exact=reference_vertices.tobytes()==candidate_vertices.tobytes(),
        full_ordered_faces=len(faces))
    path=output/'byte_sections.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','default-repeat','verbose-repeat','brain','output'):p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
