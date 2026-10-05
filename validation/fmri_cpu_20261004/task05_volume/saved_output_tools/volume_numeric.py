"""Pure saved-output numeric checks; no FNIT API, registrations or resampling."""
import hashlib
import gzip
import struct
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def raw_header_sha(path):
    # nib.load consumes the on-disk scl_slope/inter into its proxy. Hash the
    # stored header as well so exact-header claims retain those original bits.
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rb') as stream:
        prefix=stream.read(4)
        sizes=[struct.unpack(order+'i',prefix)[0] for order in ('<','>')]
        size=next((n for n in sizes if n in (348,540)),None)
        if size is None:raise ValueError('Unknown saved NIfTI header size')
        block=prefix+stream.read(size-4)
    if len(block)!=size:raise ValueError('Truncated saved NIfTI header')
    return hashlib.sha256(block).hexdigest()


def compare_arrays(first, second):
    import numpy as np
    a, b = np.asanyarray(first), np.asanyarray(second)
    if a.shape != b.shape:
        return {'shape_equal': False, 'accepted': False}
    d = a.astype(np.float64) - b.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    count = int(a.size)
    return {'shape_equal': True, 'dtype_equal': a.dtype == b.dtype,
            'finite': finite, 'values': count, 'different_values': int(np.count_nonzero(a != b)),
            'max_abs': float(np.max(np.abs(d), initial=0)) if finite else None,
            'rmse': float(np.sqrt(np.sum(d*d, dtype=np.float64)/count)) if finite and count else None,
            'accepted': finite and a.dtype == b.dtype and bool(np.array_equal(a, b))}


def compare_nifti(first, second):
    import nibabel as nib
    import numpy as np
    a, b = nib.load(str(first), keep_file_open=True), nib.load(str(second), keep_file_open=True)
    geometry = {'shape_equal': a.shape == b.shape, 'dtype_equal': a.get_data_dtype() == b.get_data_dtype(),
                'affine_exact': bool(np.array_equal(a.affine, b.affine)),
                'qform_exact': bool(np.array_equal(a.get_qform(), b.get_qform())),
                'sform_exact': bool(np.array_equal(a.get_sform(), b.get_sform())),
                'qform_code_equal': int(a.header['qform_code']) == int(b.header['qform_code']),
                'sform_code_equal': int(a.header['sform_code']) == int(b.header['sform_code']),
                'binary_header_exact': a.header.binaryblock == b.header.binaryblock,
                'stored_header_exact': raw_header_sha(first)==raw_header_sha(second)}
    if not geometry['shape_equal']:
        return dict(geometry, accepted=False)
    chunks = ([(slice(None),)*(len(a.shape)-1)+(slice(start, start+8),) for start in range(0, a.shape[-1], 8)]
              if len(a.shape) == 4 else [Ellipsis])
    size = different = 0
    square = maximum = 0.
    finite = True
    for selection in chunks:
        x, y = np.asanyarray(a.dataobj[selection]), np.asanyarray(b.dataobj[selection])
        valid = bool(np.isfinite(x).all() and np.isfinite(y).all())
        finite = finite and valid
        size += x.size
        different += int(np.count_nonzero(x != y))
        if valid:
            delta = x.astype(np.float64) - y.astype(np.float64)
            square += float(np.sum(delta*delta, dtype=np.float64))
            maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
    return dict(geometry, finite=finite, values=int(size), different_values=different,
                max_abs=maximum if finite else None,
                rmse=(square/size)**.5 if finite and size else None,
                accepted=all(geometry.values()) and finite and different == 0)


def physical_pair(candidate, official, *, frames=None, mask=False):
    """Full values on the same physical lattice; only lossless axis flips/permutations."""
    import nibabel as nib
    import numpy as np
    a,b=nib.load(str(candidate)),nib.load(str(official))
    scales={'mm':1.,'meter':1000.,'micron':.001}
    units=[x.header.get_xyzt_units() for x in (a,b)]
    row={'candidate_sha256':sha(candidate),'official_sha256':sha(official),
         'candidate_shape':list(a.shape),'official_shape':list(b.shape),
         'candidate_dtype':str(a.get_data_dtype()),'official_dtype':str(b.get_data_dtype()),
         'units':units,'comparison_scope':'All values/frames; lossless spatial axis-index alignment only. No interpolation,registration,intensity fitting or FNITclean comparison.'}
    if any(unit[0] not in scales for unit in units):
        return dict(row,status='not_comparable_spatial_units_unproved')
    affines=[]
    for image,unit in zip((a,b),units):
        matrix=image.affine.copy();matrix[:3,:]*=scales[unit[0]];affines.append(matrix)
    transform=nib.orientations.ornt_transform(nib.orientations.io_orientation(affines[1]),
                                               nib.orientations.io_orientation(affines[0]))
    aligned=affines[1]@nib.orientations.inv_ornt_aff(transform,b.shape[:3])
    aligned_shape=tuple(b.shape[int(k)] for k in np.argsort(transform[:,0]))+b.shape[3:]
    row.update(axis_permutation_flip=transform.tolist(),candidate_axis_codes=list(nib.aff2axcodes(affines[0])),
               official_axis_codes=list(nib.aff2axcodes(affines[1])),
               affine_max_abs_mm=float(np.max(np.abs(affines[0]-aligned))))
    if a.shape!=aligned_shape or not np.allclose(affines[0],aligned,rtol=0,atol=1e-5):
        return dict(row,status='not_comparable_distinct_physical_lattices')
    if frames is not None:
        times={'sec':1.,'msec':.001,'usec':1e-6}
        if (a.ndim!=4 or b.ndim!=4 or a.shape[3]!=frames
                or any(u[1] not in times for u in units)):
            return dict(row,status='not_comparable_complete_time_axis_unproved')
        tr=[float(x.header.get_zooms()[3])*times[u[1]] for x,u in zip((a,b),units)]
        row['TR_seconds']=tr
        if not np.isclose(tr[0],tr[1],rtol=0,atol=1e-6):
            return dict(row,status='not_comparable_time_spacing')
    sums=np.zeros(5,dtype=np.float64);square=maximum=0.;count=diff=0;finite=True
    temporal=[np.zeros(a.shape[:3],dtype=np.float64) for _ in range(5)] if frames is not None else None
    intersection=positive_a=positive_b=0
    starts=range(0,a.shape[3],8) if a.ndim==4 else [None]
    for start in starts:
        selection=(Ellipsis,slice(start,start+8)) if start is not None else Ellipsis
        x=np.asanyarray(a.dataobj[selection]);y=np.asanyarray(b.dataobj[selection])
        y=nib.orientations.apply_orientation(y,transform)
        valid=bool(np.isfinite(x).all() and np.isfinite(y).all());finite=finite and valid
        count+=x.size;diff+=int(np.count_nonzero(x!=y))
        if not valid:continue
        x=x.astype(np.float64);y=y.astype(np.float64);delta=x-y
        square+=float(np.sum(delta*delta));maximum=max(maximum,float(np.max(np.abs(delta),initial=0)))
        sums+=np.array([np.sum(x),np.sum(y),np.sum(x*x),np.sum(y*y),np.sum(x*y)])
        if temporal is not None:
            for destination,value in zip(temporal,(x,y,x*x,y*y,x*y)):
                destination+=np.sum(value,axis=3)
        if mask:
            ma=x>0;mb=y>0;positive_a+=int(ma.sum());positive_b+=int(mb.sum());intersection+=int((ma&mb).sum())
    row.update(status='compared_same_physical_lattice' if finite else 'failed_nonfinite',
               values=int(count),different_values=diff,all_finite=finite,
               rmse=(square/count)**.5 if finite and count else None,max_abs=maximum if finite else None,
               decoded_values_exact=finite and diff==0)
    if finite and count:
        covariance=sums[4]-sums[0]*sums[1]/count
        va=max(0.,sums[2]-sums[0]**2/count);vb=max(0.,sums[3]-sums[1]**2/count)
        row['full_lattice_pearson']=float(covariance/np.sqrt(va*vb)) if va>0 and vb>0 else None
    if temporal is not None and finite:
        sx,sy,sxx,syy,sxy=temporal;va=sxx-sx*sx/frames;vb=syy-sy*sy/frames
        varying=(va>np.finfo(np.float64).eps*np.maximum(sxx,1.))&(vb>np.finfo(np.float64).eps*np.maximum(syy,1.))
        corr=(sxy[varying]-sx[varying]*sy[varying]/frames)/np.sqrt(va[varying]*vb[varying])
        row['temporal_correlation']={'mean_varying_voxels':float(corr.mean()) if corr.size else None,
                                     'varying_voxels':int(varying.sum()),'constant_series_excluded':int(varying.size-varying.sum()),
                                     'rmse_scope':'RMSE retains every voxel including constant/zero series.'}
    if mask:row['dice']=2.*intersection/(positive_a+positive_b) if positive_a+positive_b else 1.
    return row


def affine_world(spec):
    """Decode a declared affine convention, without calling any imaging software."""
    import nibabel as nib
    import numpy as np
    path=Path(spec['path'])
    if sha(path)!=spec['sha256']:raise ValueError('Pinned affine changed')
    convention=spec['convention']
    if convention=='itk_lps_push':
        text=path.read_text().splitlines()
        transforms=[line for line in text if line.startswith('Transform:')]
        if len(transforms)!=1 or 'AffineTransform_double_3_3' not in transforms[0]:
            raise ValueError('Only one proven ITK double3D affine is supported')
        parameters=np.fromstring(next(x.split(':',1)[1] for x in text if x.startswith('Parameters:')),sep=' ')
        center=np.fromstring(next(x.split(':',1)[1] for x in text if x.startswith('FixedParameters:')),sep=' ')
        if parameters.size!=12 or center.size!=3:raise ValueError('ITK affine parameters invalid')
        matrix=np.eye(4);matrix[:3,:3]=parameters[:9].reshape(3,3)
        matrix[:3,3]=parameters[9:]+center-matrix[:3,:3]@center
        ras=np.diag([-1.,-1.,1.,1.]);matrix=ras@matrix@ras
    elif convention=='fsl_scaled_mm_push':
        matrix=np.loadtxt(path)
        frames=[]
        for label in ('moving_image','fixed_image'):
            info=spec[label]
            if sha(info['path'])!=info['sha256']:raise ValueError('FLIRT coordinate input grid changed')
            image=nib.load(info['path'])
            if image.header.get_xyzt_units()[0]!='mm':raise ValueError('Prove FLIRT grid units in mm')
            scaled=np.eye(4);zooms=np.asarray(image.header.get_zooms()[:3]);scaled[:3,:3]=np.diag(zooms)
            if np.linalg.det(image.affine[:3,:3])>0:
                scaled[0,0]*=-1;scaled[0,3]=(image.shape[0]-1)*zooms[0]
            frames.append((image.affine,scaled))
        moving,fixed=frames
        matrix=fixed[0]@np.linalg.inv(fixed[1])@matrix@moving[1]@np.linalg.inv(moving[0])
    elif convention=='world_ras_push':matrix=np.loadtxt(path)
    else:raise ValueError('Unproved affine convention; nonlinear transforms are not comparable')
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Invalid world affine')
    return matrix

