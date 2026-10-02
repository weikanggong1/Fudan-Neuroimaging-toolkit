from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from .config import FSL2111Config, FSL_EDDY_COMMIT, FSL_EDDY_VERSION
from .geometry import (
    apply_slm_linear,
    fsl_rotation_matrix,
    identity_grid,
    load_topup_movpar,
    matrix_to_movepar,
    movepar_to_matrix,
    quadratic_ec_basis,
    rereference_movement,
    separate_offset_from_movement,
)
from .gp import NewSphericalGP, _shell_groups
from .outlier import detect_slice_outliers
from .registration import parameter_update
from .shell_alignment import register_shell_mean, update_shell_movements
from .spline import fsl_cubic_coefficients, _pad_cubic_coefficients
from .warp import model_to_scan, sample_linear_mask, unwarp_scan_to_model

# Reuse only the already validated FNIT TOPUP coefficient decoder and gradient IO.
from ..topup_field import _load_topup_field
from ..._dmri import configure_device, load_bvals, load_bvecs, image_like, output_path


@dataclass
class StrictEDDYResult:
    corrected: nib.Nifti1Image
    rotated_bvecs: np.ndarray
    parameters: np.ndarray
    movement_rms: np.ndarray
    restricted_movement_rms: np.ndarray
    outlier_map: np.ndarray
    outlier_n_stdev_map: np.ndarray
    outlier_n_sqr_stdev_map: np.ndarray
    outlier_report_lines: list[str]
    qc: dict

    def save(self, out, overwrite=False):
        root=Path(out)
        paths={
            output_path(root): self.corrected,
            root.with_name(root.name+'.eddy_rotated_bvecs'): self.rotated_bvecs,
            root.with_name(root.name+'.eddy_parameters'): self.parameters,
            root.with_name(root.name+'.eddy_movement_rms'): self.movement_rms,
            root.with_name(root.name+'.eddy_restricted_movement_rms'): self.restricted_movement_rms,
            root.with_name(root.name+'.eddy_outlier_map'): ('One row per scan, one column per slice. Outlier: 1, Non-outlier: 0', self.outlier_map),
            root.with_name(root.name+'.eddy_outlier_n_stdev_map'): ('One row per scan, one column per slice. b0s set to zero', self.outlier_n_stdev_map),
            root.with_name(root.name+'.eddy_outlier_n_sqr_stdev_map'): ('One row per scan, one column per slice. b0s set to zero', self.outlier_n_sqr_stdev_map),
            root.with_name(root.name+'.eddy_outlier_report'): self.outlier_report_lines,
            root.with_name(root.name+'.eddy_qc.json'): self.qc,
        }
        for p in paths:
            if p.exists() and not overwrite: raise FileExistsError(p)
        root.parent.mkdir(parents=True,exist_ok=True)
        for p,v in paths.items():
            if isinstance(v,nib.spatialimages.SpatialImage): nib.save(v,str(p))
            elif isinstance(v,dict): p.write_text(json.dumps(v,indent=2)+'\n')
            elif isinstance(v,list): p.write_text('\n'.join(v)+('\n' if v else ''))
            elif isinstance(v,tuple):
                header,a=v
                np.savetxt(p,a,fmt='%.10g',header=header,comments='')
            else: np.savetxt(p,v,fmt='%.10g')
        return paths


def _init_movement_from_topup(topup, indices, acquisition, shape, voxel_sizes, device):
    mp=np.asarray(load_topup_movpar(topup, len(acquisition)),dtype=np.float64)
    rows=mp[indices]
    t=torch.as_tensor(rows,dtype=torch.float64,device=device)
    M=movepar_to_matrix(t,shape,voxel_sizes)
    # ECScanManager initialises with Matrix2MovePar(inverse(TOPUP forward matrix)).
    return matrix_to_movepar(torch.linalg.inv(M),shape,voxel_sizes).to(torch.float32)


def _interpolate_b0_movement(movement, b0_global, dwi_global):
    if len(b0_global) < 2: return movement
    out=movement.clone(); b0=np.asarray(b0_global)
    for d in dwi_global:
        pos=np.searchsorted(b0,d)
        if pos==0: out[d]=out[int(b0[0])]
        elif pos==len(b0): out[d]=out[int(b0[-1])]
        else:
            l,r=int(b0[pos-1]),int(b0[pos]); w=(d-l)/(r-l)
            out[d]=(1-w)*out[l]+w*out[r]
    return out


def _shared_transform_pe_axis(phase_np, validated_axis):
    """Use a CPU-known axis only when it matches the legacy float32 selector."""
    active=np.abs(np.asarray(phase_np,dtype=np.float32)) > np.float32(1e-8)
    matches=active[:,validated_axis] & ~active[:,:validated_axis].any(1)
    return validated_axis if bool(matches.all()) else None


def _unwarp_many(work, params, susceptibility, phase, readout, voxel_sizes, precision, indices, *,
                 grid=None, basis=None, pe_axis=None):
    imgs=[]; masks=[]
    for gi in indices:
        y,m,_,_=unwarp_scan_to_model(work[gi],params[gi,:6],params[gi,6:],susceptibility,
                                     phase[gi],readout[gi],voxel_sizes,precision,True,
                                     grid=grid,basis=basis,pe_axis=pe_axis)
        imgs.append(y); masks.append(m)
    return torch.stack(imgs), torch.stack(masks)


def _movement_displacement_rms(movement, mask, shape, voxel_sizes, pe_axis=None):
    # FSL's RMS is mean voxel displacement in the actual brain mask.
    device=movement.device; dtype=movement.dtype
    axes=[torch.arange(n,device=device,dtype=dtype) for n in shape]
    g=torch.stack(torch.meshgrid(*axes,indexing='ij'))
    vs=torch.as_tensor(voxel_sizes,device=device,dtype=dtype)[:,None,None,None]
    p=(g*vs).reshape(3,-1)
    m=mask.reshape(-1)
    vals=[]
    for mp in movement:
        if pe_axis is not None:
            mp=mp.clone()
            mp[pe_axis]=0
        M=movepar_to_matrix(mp[None],shape,voxel_sizes)[0]
        h=torch.cat((p,torch.ones((1,p.shape[1]),device=device,dtype=dtype)),0)
        q=(M@h)[:3]
        disp=q-p
        vals.append(disp[:,m])
    fields=vals
    absr=[]; rel=[torch.tensor(0.,device=device,dtype=dtype)]
    for i,d in enumerate(fields):
        absr.append(torch.sqrt((d*d).sum(0).mean()))
        if i: rel.append(torch.sqrt(((d-fields[i-1])**2).sum(0).mean()))
    return torch.stack((torch.stack(absr),torch.stack(rel)),1)


class TorchEDDYFSL2111:
    """Source-aligned PyTorch/CUDA implementation of the UKB eddy 2111 path.

    The class deliberately contains no Adam optimiser and no image pyramid.
    It follows eddy.cpp's alternating predictor/outlier/Gauss-Newton loop.
    """
    def __init__(self,device='cuda:0',config: FSL2111Config|None=None):
        self.device=configure_device(device); self.config=config or FSL2111Config()
        if self.device.type=='cuda':
            torch.backends.cuda.matmul.allow_tf32=bool(self.config.use_tf32)

    def __call__(self,imain,mask,acqp,index,bvecs,bvals,*,topup=None,ref_scan_no=None,gp_seed=None):
        cfg=self.config; ref_scan_no=cfg.ref_scan_no if ref_scan_no is None else int(ref_scan_no)
        nim=nib.load(os.fspath(imain)); raw_np=np.asarray(nim.dataobj,dtype=np.float32)
        mask_image=nib.load(os.fspath(mask)); mask_np=np.asarray(mask_image.dataobj)>0
        bv=load_bvals(bvals); bg=load_bvecs(bvecs,len(bv)).T
        acq=np.loadtxt(acqp,dtype=np.float64,ndmin=2); ind=np.loadtxt(index,dtype=int).reshape(-1)-1
        if raw_np.ndim!=4 or raw_np.shape[3]!=len(bv): raise ValueError('input dimension mismatch')
        if mask_np.shape!=raw_np.shape[:3]: raise ValueError('mask and DWI grids have different shapes')
        if not np.allclose(mask_image.affine,nim.affine,rtol=0,atol=1e-5): raise ValueError('mask and DWI grids have different affines')
        if len(ind)!=len(bv): raise ValueError('index length mismatch')
        if acq.shape[1]!=4 or np.any(ind<0) or np.any(ind>=len(acq)): raise ValueError('invalid acqp or index')
        if ref_scan_no<0 or ref_scan_no>=len(bv): raise ValueError('ref_scan_no is outside the DWI volume range')
        phase_np=acq[ind,:3]; axes=np.flatnonzero(np.any(np.abs(phase_np)>1e-6,axis=0))
        if len(axes)!=1: raise NotImplementedError('pinned UKB strict backend requires one shared PE axis')
        pe_axis=int(axes[0]); voxel_sizes=tuple(float(x) for x in nim.header.get_zooms()[:3])
        device=self.device
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        mask_t=torch.as_tensor(mask_np,device=device)
        raw=torch.as_tensor(np.moveaxis(raw_np,-1,0).copy(),dtype=torch.float32,device=device)
        bvals_t=torch.as_tensor(bv,dtype=torch.float32,device=device)
        bvec_t=torch.as_tensor(bg,dtype=torch.float32,device=device)
        phase=torch.as_tensor(phase_np,dtype=torch.float32,device=device)
        readout=torch.as_tensor(acq[ind,3],dtype=torch.float32,device=device)
        # TOPUP coefficients need full float32 matmul precision. TF32 truncates
        # the field by up to 0.17 Hz on the reference AP scan.
        tf32=torch.backends.cuda.matmul.allow_tf32 if device.type=='cuda' else None
        try:
            if device.type=='cuda': torch.backends.cuda.matmul.allow_tf32=False
            susceptibility,_,_=_load_topup_field(topup,raw_np.shape[:3],device,pe_axis)
        finally:
            if device.type=='cuda': torch.backends.cuda.matmul.allow_tf32=tf32
        # These tensors depend only on this run's fixed grid and decoded TOPUP
        # field. Work images and GP predictions are intentionally not cached.
        warp_geometry={
            'grid':identity_grid(raw_np.shape[:3],device,raw.dtype),
            'basis':quadratic_ec_basis(raw_np.shape[:3],voxel_sizes,device,raw.dtype),
            # Input validation uses 1e-6, whereas the legacy transform uses
            # float32 1e-8. Retain its fallback for noncanonical accepted rows.
            'pe_axis':_shared_transform_pe_axis(phase_np,pe_axis),
        }
        susceptibility_coeff=fsl_cubic_coefficients(susceptibility,cfg.spline_precision)
        model_constants={
            **warp_geometry,
            'susc_coeff':susceptibility_coeff,
            'susc_padded_coeff':_pad_cubic_coefficients(susceptibility_coeff[None],'mirror'),
        }
        # Match ECScanManager's internal intensity scaling.
        b0_global=np.flatnonzero(bv < cfg.b0_threshold).tolist(); dwi_global=np.flatnonzero(bv >= cfg.b0_threshold).tolist()
        if not b0_global or not dwi_global: raise ValueError('strict eddy path requires both b0 and DWI scans')
        scale=100.0/float(raw[b0_global[0]][mask_t].mean().item())
        original=raw*scale; work=original.clone()
        movement=_init_movement_from_topup(topup,ind,acq,raw_np.shape[:3],voxel_sizes,device)
        params=torch.zeros((len(bv),16),device=device,dtype=torch.float32); params[:,:6]=movement
        dwi_local_b=bvals_t[dwi_global]; dwi_local_g=bvec_t[dwi_global]
        old_outliers=torch.zeros((len(dwi_global),raw_np.shape[2]),dtype=torch.bool,device=device)
        nsv_final=torch.zeros_like(old_outliers,dtype=torch.float32); nsq_final=torch.zeros_like(nsv_final)
        iter_log=[]; started=time.perf_counter()
        # FSL DoVolumeToVolumeRegistration runs the complete b0 Register first.
        if len(b0_global)>1:
            for it,fwhm in enumerate(cfg.fwhm_mm):
                ub0,mb0=_unwarp_many(work,params,susceptibility,phase,readout,voxel_sizes,cfg.spline_precision,b0_global,**warp_geometry)
                b0pred=ub0.mean(0)
                b0_fov=mb0.all(0)
                b0log=[]
                for gi in b0_global:
                    pnew,diag=parameter_update(b0pred,work[gi],params[gi],susceptibility,phase[gi],readout[gi],
                                               voxel_sizes,float(fwhm),b0_fov,cfg.spline_precision,active_indices=range(6),**model_constants)
                    params[gi]=pnew; b0log.append(diag)
                iter_log.append({'stage':'b0','iteration':it,'fwhm_mm':float(fwhm),
                                 'mean_update_mss':float(np.mean([x['mss'] for x in b0log]))})
                print(f'EDDY b0 iteration {it + 1}/{cfg.niter}: {time.perf_counter() - started:.1f}s',flush=True)
            # ApplyB0LocationReference after all b0 iterations.
            params[b0_global,:6]=rereference_movement(params[b0_global,:6],0,raw_np.shape[:3],voxel_sizes)
            # If b0s are interspersed, FSL linearly inter/extrapolates their movement
            # to initialise the DWI scans exactly once before DWI Register.
            if (len(b0_global)>2 and b0_global[0]<0.25*len(bv)
                    and b0_global[-1]>0.75*len(bv)):
                params[:,:6]=_interpolate_b0_movement(params[:,:6],b0_global,dwi_global)
        # Complete DWI Register, with one GP/outlier/update cycle per FWHM iteration.
        for it,fwhm in enumerate(cfg.fwhm_mm):
            udwi,vmasks=_unwarp_many(work,params,susceptibility,phase,readout,voxel_sizes,cfg.spline_precision,dwi_global,**warp_geometry)
            common_fov=vmasks.all(0)
            common_mask=mask_t & common_fov
            # FSL DataSelector calls srand(initrand) on every construction when
            # --initrand is non-zero, so a user-supplied seed is reused verbatim
            # for every GP rebuild rather than advanced by outer iteration.
            seed_for_iter=(None if gp_seed is None else int(gp_seed))
            gp=NewSphericalGP(dwi_local_b,dwi_local_g,cfg.shell_tolerance,cfg.ff,cfg.gp_nm_maxiter)
            gp.fit(udwi,common_mask,cfg.nvoxhp,seed_for_iter,float(fwhm),voxel_sizes)
            pred_obs=[]; pred_masks=[]
            for li,gi in enumerate(dwi_global):
                po,pm,_,coords=model_to_scan(gp.predict(li,False),params[gi,:6],params[gi,6:],susceptibility,
                                        phase[gi],readout[gi],voxel_sizes,cfg.spline_precision,True,**model_constants)
                pred_obs.append(po); pred_masks.append(pm & sample_linear_mask(common_mask,coords))
            ol=detect_slice_outliers(original[dwi_global],torch.stack(pred_obs),torch.stack(pred_masks),old_outliers,
                                     cfg.ol_nstd,cfg.ol_nvox,1,False,False)
            old_outliers=ol.outlier_map.clone(); nsv_final=ol.n_stdev; nsq_final=ol.n_sqr_stdev
            # Register() replaces only from the second DWI iteration onward and then reloads GP.
            if it>0 and old_outliers.any():
                for li,gi in enumerate(dwi_global):
                    if not old_outliers[li].any(): continue
                    po,pm,_,coords=model_to_scan(gp.predict(li,True),params[gi,:6],params[gi,6:],susceptibility,
                                            phase[gi],readout[gi],voxel_sizes,cfg.spline_precision,True,**model_constants)
                    for z in torch.nonzero(old_outliers[li],as_tuple=False).flatten().tolist():
                        mm=(pm & sample_linear_mask(common_mask,coords,threshold=0.9))[:,:,z]
                        work[gi,:,:,z]=torch.where(mm,po[:,:,z],work[gi,:,:,z])
                udwi,vmasks=_unwarp_many(work,params,susceptibility,phase,readout,voxel_sizes,cfg.spline_precision,dwi_global,**warp_geometry)
                common_fov=vmasks.all(0)
                common_mask=mask_t & common_fov
                gp=NewSphericalGP(dwi_local_b,dwi_local_g,cfg.shell_tolerance,cfg.ff,cfg.gp_nm_maxiter)
                gp.fit(udwi,common_mask,cfg.nvoxhp,seed_for_iter,float(fwhm),voxel_sizes)
            update_log=[]
            # Predictor is held fixed within this official iteration; each scan receives one GN update.
            for li,gi in enumerate(dwi_global):
                pnew,diag=parameter_update(gp.predict(li,False),work[gi],params[gi],susceptibility,phase[gi],readout[gi],
                                           voxel_sizes,float(fwhm),common_fov,cfg.spline_precision,active_indices=range(16),**model_constants)
                params[gi]=pnew; update_log.append(diag)
            params[dwi_global,6:]=apply_slm_linear(params[dwi_global,6:],dwi_local_b,dwi_local_g)
            m,e=separate_offset_from_movement(params[dwi_global,:6],params[dwi_global,6:],dwi_local_b,dwi_local_g,
                                              phase[dwi_global],readout[dwi_global],voxel_sizes)
            params[dwi_global,:6]=m; params[dwi_global,6:]=e
            # SeparateFieldOffsetFromMovement itself reapplies the DWI and overall
            # location references in eddy 2111.0; this affects the next GP iteration.
            params[dwi_global,:6]=rereference_movement(params[dwi_global,:6],0,raw_np.shape[:3],voxel_sizes)
            params[:,:6]=rereference_movement(params[:,:6],ref_scan_no,raw_np.shape[:3],voxel_sizes)
            iter_log.append({'stage':'dwi','iteration':it,'fwhm_mm':float(fwhm),'gp_seed':getattr(gp,'data_selector_seed',None),
                             'outliers':int(old_outliers.sum().item()),'mean_update_mss':float(np.mean([x['mss'] for x in update_log]))})
            print(f'EDDY DWI iteration {it + 1}/{cfg.niter}: {time.perf_counter() - started:.1f}s',flush=True)
        # ApplyDWILocationReference, followed by the overall requested location reference.
        params[dwi_global,:6]=rereference_movement(params[dwi_global,:6],0,raw_np.shape[:3],voxel_sizes)
        params[:,:6]=rereference_movement(params[:,:6],ref_scan_no,raw_np.shape[:3],voxel_sizes)
        # FSL 2111 defaults to full six-parameter post-eddy alignment; the PE-only
        # alignment is calculated for a report but is not applied.
        if cfg.enable_post_eddy_shell_alignment:
            corrected_pre=[]; valid_pre=[]
            for gi in range(len(bv)):
                y,m,_,_=unwarp_scan_to_model(work[gi],params[gi,:6],params[gi,6:],susceptibility,phase[gi],readout[gi],voxel_sizes,cfg.spline_precision,True,**warp_geometry)
                corrected_pre.append(y); valid_pre.append(m)
            corrected_pre=torch.stack(corrected_pre); valid_pre=torch.stack(valid_pre)
            b0mean=corrected_pre[b0_global].mean(0)
            groups,_,_= _shell_groups(dwi_local_b,cfg.shell_tolerance)
            shell_globals=[[dwi_global[x] for x in g] for g in groups]
            peas_mask=valid_pre.all(0)
            updates=[register_shell_mean(b0mean,corrected_pre[g].mean(0),peas_mask,voxel_sizes)
                     for g in shell_globals]
            params[:,:6]=update_shell_movements(params[:,:6],shell_globals,updates,raw_np.shape[:3],voxel_sizes)
        # eddy.cpp temporarily sets ff=1 for FinalOLCheck, then restores ff.
        udwi,vmasks=_unwarp_many(work,params,susceptibility,phase,readout,voxel_sizes,cfg.spline_precision,dwi_global,**warp_geometry)
        common_mask=mask_t & vmasks.all(0)
        gp=NewSphericalGP(dwi_local_b,dwi_local_g,cfg.shell_tolerance,1.0,cfg.gp_nm_maxiter)
        gp.fit(udwi,common_mask,cfg.nvoxhp,gp_seed,0.0,voxel_sizes)
        pred_obs=[]; pred_masks=[]
        for li,gi in enumerate(dwi_global):
            po,pm,_,coords=model_to_scan(gp.predict(li,False),params[gi,:6],params[gi,6:],susceptibility,phase[gi],readout[gi],voxel_sizes,cfg.spline_precision,True,**model_constants)
            pred_obs.append(po); pred_masks.append(pm & sample_linear_mask(common_mask,coords))
        ol=detect_slice_outliers(original[dwi_global],torch.stack(pred_obs),torch.stack(pred_masks),old_outliers,cfg.ol_nstd,cfg.ol_nvox)
        old_outliers=ol.outlier_map; nsv_final=ol.n_stdev; nsq_final=ol.n_sqr_stdev
        for li,gi in enumerate(dwi_global):
            if not old_outliers[li].any(): continue
            po,pm,_,coords=model_to_scan(gp.predict(li,True),params[gi,:6],params[gi,6:],susceptibility,phase[gi],readout[gi],voxel_sizes,cfg.spline_precision,True,**model_constants)
            for z in torch.nonzero(old_outliers[li],as_tuple=False).flatten().tolist():
                mm=(pm & sample_linear_mask(common_mask,coords))[:,:,z]
                work[gi,:,:,z]=torch.where(mm,po[:,:,z],work[gi,:,:,z])
        # Final Jacobian resampling, and intersection output mask as FSL default.
        outs=[]; oms=[]
        for gi in range(len(bv)):
            y,m,_,_=unwarp_scan_to_model(work[gi],params[gi,:6],params[gi,6:],susceptibility,phase[gi],readout[gi],voxel_sizes,cfg.spline_precision,True,
                                         pe_extrapolation_valid=True,**warp_geometry)
            outs.append(y/scale); oms.append(m)
        out=torch.stack(outs); om=torch.stack(oms).all(0); out=out*om
        # Rotated b-vectors use inverse movement rotation.
        R=fsl_rotation_matrix(params[:,:6][:,3:6]); Rin=R.transpose(1,2)
        rg=torch.bmm(Rin,bvec_t[:,:,None]).squeeze(-1); rg=rg/rg.norm(dim=1,keepdim=True).clamp_min(1e-12)
        # Repack outlier matrices with b0 rows zero.
        outmap=torch.zeros((len(bv),raw_np.shape[2]),dtype=torch.int32,device=device)
        nsv=torch.zeros((len(bv),raw_np.shape[2]),dtype=torch.float32,device=device)
        nsq=torch.zeros_like(nsv)
        outmap[dwi_global]=old_outliers.int(); nsv[dwi_global]=nsv_final; nsq[dwi_global]=nsq_final
        report=[]
        for li,gi in enumerate(dwi_global):
            for z in torch.nonzero(old_outliers[li],as_tuple=False).flatten().tolist():
                report.append(f'Slice {z} in scan {gi} is an outlier with mean {float(nsv_final[li,z]):.10g} standard deviations off, and mean squared {float(nsq_final[li,z]):.10g} standard deviations off.')
        rms=_movement_displacement_rms(params[:,:6],mask_t,raw_np.shape[:3],voxel_sizes,None)
        rrms=_movement_displacement_rms(params[:,:6],mask_t,raw_np.shape[:3],voxel_sizes,pe_axis)
        elapsed=time.perf_counter()-started
        qc={'backend':'fsl2111_strict','fsl_eddy_version':FSL_EDDY_VERSION,'fsl_eddy_commit':FSL_EDDY_COMMIT,
            'device':str(device),'tf32':cfg.use_tf32,'elapsed_seconds':elapsed,'iterations':iter_log,
            'gp_seed_override':gp_seed,'peak_cuda_memory_bytes':(torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)}
        return StrictEDDYResult(image_like(np.moveaxis(out.detach().cpu().numpy(),0,-1),nim),rg.T.detach().cpu().numpy(),
                                params.detach().cpu().numpy(),rms.detach().cpu().numpy(),rrms.detach().cpu().numpy(),
                                outmap.detach().cpu().numpy(),nsv.detach().cpu().numpy(),nsq.detach().cpu().numpy(),report,qc)

    def run(self,*args,out,overwrite=False,**kwargs):
        r=self(*args,**kwargs); r.save(out,overwrite=overwrite); return r
