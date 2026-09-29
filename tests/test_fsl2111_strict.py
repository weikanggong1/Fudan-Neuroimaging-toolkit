import math
import numpy as np
import torch

from fnit.eddy.fsl2111_strict.geometry import (
    movepar_to_matrix, matrix_to_movepar, quadratic_ec_basis,
    apply_slm_linear, rereference_movement, jacobian_from_pe_displacement, identity_grid,
)
from fnit.eddy.fsl2111_strict.spline import fsl_cubic_coefficients, sample_cubic_periodic, sample_cubic_periodic_fast
from fnit.eddy.fsl2111_strict.outlier import detect_slice_outliers


def test_movepar_matrix_roundtrip():
    mp=torch.tensor([[1.2,-0.8,0.4,0.01,-0.02,0.03]],dtype=torch.float64)
    M=movepar_to_matrix(mp,(104,104,72),(2.0,2.0,2.0))
    got=matrix_to_movepar(M,(104,104,72),(2.0,2.0,2.0))
    assert torch.allclose(got,mp,atol=1e-10,rtol=0)


def test_reference_scan_becomes_identity():
    mp=torch.tensor([[1.,0,0,0,0,0],[2.,0,0,0,0,0]],dtype=torch.float64)
    got=rereference_movement(mp,1,(20,20,20),(2,2,2))
    assert torch.allclose(got[1],torch.zeros(6,dtype=got.dtype),atol=1e-10)


def test_quadratic_basis_order():
    b=quadratic_ec_basis((3,3,3),(1,1,1),'cpu',torch.float64)
    # centre voxel: all spatial terms zero, constant one
    assert torch.allclose(b[:9,1,1,1],torch.zeros(9,dtype=torch.float64))
    assert b[9,1,1,1] == 1


def test_periodic_cubic_reproduces_voxel_centres():
    torch.manual_seed(1)
    x=torch.randn(7,8,9)
    c=fsl_cubic_coefficients(x)
    a=[torch.arange(n,dtype=x.dtype) for n in x.shape]
    grid=torch.stack(torch.meshgrid(*a,indexing='ij'))[None]
    y=sample_cubic_periodic(c[None],grid)[0]
    # FSL uses a finite-precision IIR edge approximation; small residuals are expected.
    assert float((x-y).abs().max()) < 1e-2


def test_scan_to_model_jacobian_matches_cubic_field_derivative():
    torch.manual_seed(12)
    field=torch.randn(5,9,5)*0.1
    disp=torch.zeros(3,5,9,5); disp[1]=field
    jac=jacobian_from_pe_displacement(disp,1)
    coeff=fsl_cubic_coefficients(field)
    grid=identity_grid(field.shape,'cpu')
    eps=0.01
    plus=grid.clone();plus[1]+=eps
    minus=grid.clone();minus[1]-=eps
    numerical=(sample_cubic_periodic(coeff,plus)[0]-sample_cubic_periodic(coeff,minus)[0])/(2*eps)
    assert torch.allclose(jac,1+numerical,atol=2e-3,rtol=2e-3)


def test_slm_keeps_field_offset_unprojected():
    torch.manual_seed(2)
    ec=torch.randn(8,10)
    b=torch.full((8,),1000.)
    g=torch.randn(8,3); g=g/g.norm(dim=1,keepdim=True)
    got=apply_slm_linear(ec,b,g)
    assert torch.equal(got[:,-1],ec[:,-1])


def test_default_outlier_is_negative_signed_residual():
    obs=torch.ones(4,5,5,3)
    pred=obs.clone(); pred[3,:,:,1]=10.0
    mask=torch.ones_like(obs,dtype=torch.bool)
    # Use lower threshold and minvox for compact unit fixture.
    st=detect_slice_outliers(obs,pred,mask,nstd=1.0,minvox=1)
    assert bool(st.outlier_map[3,1])

from fnit.eddy.fsl2111_strict.warp import unwarp_scan_to_model, model_to_scan, _inverse_1d_displacement, sample_linear_mask
from fnit.eddy.fsl2111_strict.spline import fsl_cubic_coefficients
from fnit.eddy.fsl2111_strict.gp import NewSphericalGP, _selected_smoothed_data, _gaussian_kernel1d_fsl
from fnit.eddy.fsl2111_strict.registration import gaussian_smooth_masked
from fnit.eddy.fsl2111_strict.shell_alignment import _soft_mi


def test_registration_smoothing_ignores_values_outside_valid_fov():
    mask=torch.zeros(9,9,9,dtype=torch.bool)
    mask[2:7,2:7,2:7]=True
    image=torch.full((9,9,9),1000.)
    image[mask]=1.
    smoothed,_=gaussian_smooth_masked(image,8.,(2.,2.,2.),mask)
    assert torch.allclose(smoothed[mask],torch.ones_like(smoothed[mask]),atol=1e-5)
    assert torch.count_nonzero(smoothed[~mask])==0


def test_soft_mi_prefers_matching_shells():
    torch.manual_seed(9)
    a=torch.rand(30,30,5)
    w=torch.ones_like(a)
    exact=_soft_mi(a,a,w,(0.,1.),(0.,1.))
    shifted=_soft_mi(a,torch.roll(a,5,0),w,(0.,1.),(0.,1.))
    assert exact>shifted


def test_zero_warp_roundtrip_like_identity():
    torch.manual_seed(3)
    x=torch.randn(8,9,7)
    mp=torch.zeros(6); ec=torch.zeros(10); susc=torch.zeros_like(x)
    pe=torch.tensor([0.,1.,0.]); ro=torch.tensor(0.05)
    y,m,_,_=unwarp_scan_to_model(x,mp,ec,susc,pe,ro,(2.,2.,2.))
    assert m[1:-1,1:-1,1:-1].all()
    assert not m[0].any()
    assert float((x-y).abs().max()) < 1e-2
    z,m2,_,_=model_to_scan(x,mp,ec,susc,pe,ro,(2.,2.,2.))
    assert m2[1:-1,2:-1,1:-1].all()
    assert not m2[:,0,:].any()
    assert float((x-z).abs().max()) < 1e-2


def test_scan_to_model_mask_intersects_ec_field_validity():
    x=torch.ones(8,9,7)
    mp=torch.zeros(6); ec=torch.zeros(10); ec[9]=20.
    pe=torch.tensor([0.,1.,0.])
    _,mask,_,_=unwarp_scan_to_model(x,mp,ec,torch.zeros_like(x),pe,
                                     torch.tensor(0.05),(2.,2.,2.))
    assert not mask[:,0,:].any()


def test_mirror_spline_sampling_reflects_outside_coordinate():
    torch.manual_seed(28)
    coeff=fsl_cubic_coefficients(torch.randn(5,6,7))
    coord=torch.tensor([0.25,2.4,3.2]).reshape(1,3,1,1,1)
    outside=coord.clone(); outside[:,0]=-0.25
    inside=sample_cubic_periodic_fast(coeff,coord,boundary='mirror')
    reflected=sample_cubic_periodic_fast(coeff,outside,boundary='mirror')
    assert torch.allclose(inside,reflected,atol=1e-5,rtol=0)


def test_cached_scan_constants_preserve_model_to_scan():
    torch.manual_seed(6)
    x=torch.randn(5,6,4)
    mp=torch.tensor([0.2,-0.1,0.1,0.01,0.0,-0.01])
    ec=torch.zeros(10); susc=torch.zeros_like(x)
    pe=torch.tensor([0.,1.,0.]); ro=torch.tensor(0.05)
    direct=model_to_scan(x,mp,ec,susc,pe,ro,(2.,2.,2.))[0]
    cached=model_to_scan(x,mp,ec,susc,pe,ro,(2.,2.,2.),pred_coeff=fsl_cubic_coefficients(x),
                         susc_coeff=fsl_cubic_coefficients(susc),grid=identity_grid(x.shape,'cpu'),
                         basis=quadratic_ec_basis(x.shape,(2.,2.,2.),'cpu',x.dtype))[0]
    assert torch.equal(direct,cached)


def test_fast_cubic_matches_direct_periodic_sampling():
    torch.manual_seed(7)
    coeff=torch.randn(2,7,8,9)
    coords=torch.randn(2,3,5,6,4)*5
    direct=sample_cubic_periodic(coeff,coords)
    fast=sample_cubic_periodic_fast(coeff,coords)
    assert torch.allclose(fast,direct,atol=2e-5,rtol=2e-5)


def test_1d_inverse_brackets_transformed_voxel_centres():
    torch.manual_seed(8)
    disp=torch.randn(5,6,4)*0.1
    inverse,valid=_inverse_1d_displacement(disp,1)
    y=torch.arange(disp.shape[1],dtype=disp.dtype)[None,:,None].expand_as(disp)
    x=y+inverse
    lo=x.floor().long().clamp(0,disp.shape[1]-2)
    hi=lo+1
    sampled=torch.gather(disp,1,lo)+(x-lo)*(torch.gather(disp,1,hi)-torch.gather(disp,1,lo))
    assert torch.allclose((x+sampled)[valid],y[valid],atol=1e-5,rtol=0)
    assert torch.count_nonzero(inverse[~valid]) == 0


def test_1d_inverse_requires_both_source_mask_endpoints():
    disp=torch.zeros(2,5,2)
    source_mask=torch.ones_like(disp,dtype=torch.bool)
    source_mask[:,2,:]=False
    inverse,valid=_inverse_1d_displacement(disp,1,source_mask)
    assert not valid[:,2:4,:].any()
    assert valid[:,1,:].all()
    assert not inverse[:,2:4,:].any()


def test_generic_inverse_uses_only_lower_source_mask_endpoint():
    disp=torch.full((1,5,1),0.6)
    source_mask=torch.ones_like(disp,dtype=torch.bool)
    source_mask[:,3,:]=False
    _,derivative_valid=_inverse_1d_displacement(disp,1,source_mask,derivative_path=True)
    inverse,generic_valid=_inverse_1d_displacement(disp,1,source_mask,derivative_path=False)
    assert not derivative_valid[0,3,0]
    assert generic_valid[0,3,0]
    assert torch.allclose(inverse[0,3,0],torch.tensor(-0.6),atol=1e-6)


def test_1d_inverse_uses_first_crossing_in_folded_line():
    disp=torch.tensor([0.,2.,-1.,-1.,0.])[None,:,None]
    inverse,valid=_inverse_1d_displacement(disp,1)
    assert valid[0,2,0]
    assert torch.allclose(inverse[0,2,0],torch.tensor(-4./3.),atol=1e-6)


def test_rigid_grid_does_not_round_small_rotations_with_tf32():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip('CUDA required for TF32 regression')
    from fnit.eddy.fsl2111_strict.warp import _rigid_forward_grid
    grid=identity_grid((32,32,16),'cuda:0')
    mp=torch.tensor([0.1,-0.2,0.3,1e-5,-2e-5,3e-5],device='cuda:0')
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        exact=_rigid_forward_grid(grid,mp,(2.,2.,2.))
        torch.backends.cuda.matmul.allow_tf32=True
        got=_rigid_forward_grid(grid,mp,(2.,2.,2.))
    finally:
        torch.backends.cuda.matmul.allow_tf32=old
    assert torch.allclose(got,exact,atol=1e-5,rtol=0)


def test_masked_inverse_jacobian_uses_one_sided_difference_at_mask_edge():
    from fnit.eddy.fsl2111_strict.warp import _masked_inverse_jacobian
    inverse=torch.tensor([0.,1.,2.,3.,4.])[None,:,None]
    mask=torch.tensor([True,True,True,False,False])[None,:,None]
    got=_masked_inverse_jacobian(inverse,mask,1)
    assert torch.allclose(got[0,:,0],torch.tensor([2.,2.,2.,1.,1.]))


def test_perturbed_inverse_keeps_template_when_bracket_fails():
    from fnit.eddy.fsl2111_strict.warp import _inverse_from_template
    disp=torch.zeros(2,5,2)
    template=torch.full_like(disp,0.25)
    mask=torch.ones_like(disp,dtype=torch.bool)
    out=_inverse_from_template(disp,1,template,mask)
    assert torch.allclose(out,template)


def test_observation_mask_is_transformed_with_prediction():
    from fnit.eddy.fsl2111_strict.geometry import identity_grid
    mask=torch.zeros(5,6,4,dtype=torch.bool); mask[1:4,1:5,1:3]=True
    grid=identity_grid(mask.shape,'cpu')
    assert torch.equal(sample_linear_mask(mask,grid),mask)
    shifted=grid.clone(); shifted[0]+=1
    expected=torch.zeros_like(mask); expected[:3]=mask[1:4]
    assert torch.equal(sample_linear_mask(mask,shifted),expected)
    near_edge=grid.clone(); near_edge[0]+=0.05
    assert not sample_linear_mask(mask,near_edge)[3,2,2]
    assert sample_linear_mask(mask,near_edge,threshold=0.9)[3,2,2]


def test_final_pe_extrapolation_keeps_eddy_output_edge_voxels():
    from fnit.eddy.fsl2111_strict.warp import unwarp_scan_to_model
    scan=torch.ones(8,8,8)
    mp=torch.zeros(6)
    ec=torch.zeros(10)
    susceptibility=torch.full_like(scan,-2.)
    phase=torch.tensor([0.,1.,0.])
    _, strict, _, _=unwarp_scan_to_model(scan,mp,ec,susceptibility,phase,torch.tensor(1.),(1.,1.,1.))
    _, final, _, _=unwarp_scan_to_model(scan,mp,ec,susceptibility,phase,torch.tensor(1.),(1.,1.,1.),
                                        pe_extrapolation_valid=True)
    assert not strict[3,1,3]
    assert final[3,1,3]
    assert not final[3,0,3]


def test_restricted_rms_excludes_pe_translation_parameter():
    from fnit.eddy.fsl2111_strict.pipeline import _movement_displacement_rms
    movement=torch.tensor([[0.,1.,0.,0.,0.,0.1]])
    mask=torch.ones(8,8,8,dtype=torch.bool)
    restricted=_movement_displacement_rms(movement,mask,mask.shape,(1.,1.,1.),pe_axis=1)
    without_pe=movement.clone(); without_pe[0,1]=0
    expected=_movement_displacement_rms(without_pe,mask,mask.shape,(1.,1.,1.))
    assert torch.allclose(restricted,expected)
    assert restricted[0,0]>0


def test_spherical_gp_shapes_small_fixture():
    # compact two-shell model; maxiter kept tiny because this is a contract test.
    b=torch.tensor([1000.,1000.,1000.,2000.,2000.,2000.])
    g=torch.tensor([[1.,0,0],[0,1.,0],[0,0,1.],[1.,1,0],[1.,0,1.],[0,1,1.]])
    g=g/g.norm(dim=1,keepdim=True)
    torch.manual_seed(4)
    data=torch.randn(6,4,4,4)
    mask=torch.ones(4,4,4,dtype=torch.bool)
    gp=NewSphericalGP(b,g,100.,10.,maxiter=2).fit(data,mask,nvox=20,seed=123,fwhm_mm=0,voxel_sizes=(2,2,2))
    assert gp.K.shape==(6,6)
    assert gp.predict(0).shape==(4,4,4)
    assert gp.predict(0,exclude=True).shape==(4,4,4)
    gp.residual_data[0,0,0,0]=0
    assert gp.predict(1)[0,0,0] != 0


def test_selected_smoothing_matches_direct_weights():
    torch.manual_seed(5)
    data=torch.randn(3,6,7,5,dtype=torch.float64)
    mask=torch.rand(6,7,5)>0.25
    coords=torch.tensor([[2,3,2],[0,0,0],[5,6,4]])
    got=_selected_smoothed_data(data,mask,coords,4.0,(2.,2.,2.))
    sigma=4.0/math.sqrt(8*math.log(2))/2
    width=int(sigma-0.001)*2+3
    kernel=_gaussian_kernel1d_fsl(sigma,width,'cpu',torch.float64)
    expected=[]
    for x,y,z in coords.tolist():
        total=torch.zeros(3,dtype=data.dtype); weight=0.
        for dx in range(width):
            for dy in range(width):
                for dz in range(width):
                    xx=x+dx-width//2; yy=y+dy-width//2; zz=z+dz-width//2
                    if 0<=xx<6 and 0<=yy<7 and 0<=zz<5 and mask[xx,yy,zz]:
                        w=kernel[dx]*kernel[dy]*kernel[dz]
                        total+=w*data[:,xx,yy,zz]; weight+=w
        expected.append(total/weight)
    assert torch.allclose(got,torch.stack(expected),atol=1e-12,rtol=0)


def test_vectorized_spherical_covariance_matches_direct_formula():
    b=torch.tensor([1000.,1000.,2000.,2000.])
    g=torch.eye(3)[torch.tensor([0,1,2,0])]
    gp=NewSphericalGP(b,g)
    hp=torch.log(torch.tensor([2.,1.5,1.,0.1,0.2],dtype=torch.float64))
    th=gp._transform_hpar(hp)
    expected=torch.zeros(4,4,dtype=torch.float64)
    for i in range(4):
        for j in range(4):
            ratio=gp.angles[i,j]/th[1]
            if ratio<1: expected[i,j]=th[0]*(1-1.5*ratio+0.5*ratio**3)
            log_b_diff=torch.log(gp.group_b[gp.group_index[i]])-torch.log(gp.group_b[gp.group_index[j]])
            expected[i,j]*=torch.exp(-log_b_diff**2/(2*th[2]**2))
            if i==j: expected[i,j]+=th[3+gp.group_index[i]]
    assert torch.allclose(gp._kernel(hp),expected,atol=1e-12,rtol=0)
