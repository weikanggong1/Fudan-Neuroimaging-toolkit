"""Source-derived MSMSulc formula/schedule regression gates, not benchmarks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from fnit.msm import MSMSulcConfig
from fnit.msm._affine import _euler_matrix, _RigidCost
from fnit.msm.msmsulc import (
    _ico, _area_weights, _face_costs, _face_layout,
    _label_samples, _normalize_sphere, _rescaled_labels, _sphere_warp,
    _unfold, _variance_normalize, _native_output_qc, _triplet_data_weights,
)
from fnit.msm._sphere_map import RadialSphereMap


def test_canonical_schedule_has_three_correlation_discrete_levels():
    config=MSMSulcConfig()
    assert config.simval == (3,2,2,2)
    assert config.iterations == (50,10,15,15)
    assert config.control_grid == (6,2,3,4)
    assert config.data_grid == (6,4,5,6)
    assert MSMSulcConfig.ssd_affine().simval == (1,2,2,2)


@pytest.mark.parametrize('kwargs',[
    {'iterations':(50.0,10,15,15)}, {'control_grid':(6,True,3,4)},
    {'regularization':(0,float('nan'),7.5,7.5)},
    {'affine_step_size':float('inf')}, {'strain_exponent':float('nan')},
])
def test_config_rejects_nonfinite_or_nonintegral_options(kwargs):
    with pytest.raises(ValueError): MSMSulcConfig(**kwargs)


def test_reads_official_config_and_refuses_changed_algorithm(tmp_path):
    config=tmp_path/'MSMSulcStrainFinalconf'
    config.write_text('--simval=3,2,2,2\n--it=50,10,15,15\n--opt=AFFINE,DISCRETE,DISCRETE,DISCRETE\n'
                      '--sigma_in=0,0,0,0\n--sigma_ref=0,0,0,0\n--dopt=HOCR\n--regoption=3\n'
                      '--VN\n--rescaleL\n--triclique\n--threads=8\n')
    assert MSMSulcConfig.from_file(config)==MSMSulcConfig()
    config.write_text(config.read_text().replace('--dopt=HOCR','--dopt=MCMC'))
    with pytest.raises(ValueError,match='unsupported'):MSMSulcConfig.from_file(config)


def test_variance_normalization_uses_sample_variance():
    actual=_variance_normalize(np.array([0.,2.,5.,8.]))
    np.testing.assert_allclose(actual,(np.array([0.,2.,5.,8.])-3.75)/np.std([0.,2.,5.,8.],ddof=1),rtol=1e-15)
    np.testing.assert_array_equal(_variance_normalize(np.ones(3)),np.zeros(3))


def test_radial_metric_weights_and_unprojected_likelihood_are_distinct():
    xyz,faces=_ico(1)
    triangle=xyz[faces[0]]
    expected=np.array([.15,.3,.55])
    planar=expected@triangle
    query=torch.tensor(planar/np.linalg.norm(planar)*100)[None,:]
    mapper=RadialSphereMap(xyz,faces,'cpu')
    ids,weights,patch=mapper.weights(query)
    assert patch.item()==0
    np.testing.assert_allclose(weights.numpy()[0],expected,atol=3e-15)
    _,direct,_=mapper.weights(query,project=False)
    independent=[]
    p=query.numpy()[0]
    for a,b in ((triangle[1],triangle[2]),(triangle[0],triangle[2]),(triangle[0],triangle[1])):
        independent.append(np.linalg.norm(np.cross(a-p,b-p)))
    independent=np.asarray(independent);independent/=independent.sum()
    np.testing.assert_allclose(direct.numpy()[0],independent,atol=3e-15)
    assert np.max(np.abs(direct.numpy()[0]-expected))>1e-4


def test_triplet_projection_uses_sorted_corners_after_face_assignment():
    import math
    # Independent scalar Point operations reproduce get_target_data's sorted
    # CP corners, not the original mesh face's corner order.
    def sub(a,b):return tuple(a[i]-b[i] for i in range(3))
    def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    def dot(a,b):return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2]
    def unit(a):
        length=math.sqrt(dot(a,a))
        return tuple(value/length for value in a)
    xyz,mesh_faces=_ico(2)
    faces=np.sort(mesh_faces[[1,13,44]],axis=1)
    points=xyz[faces].mean(1)
    points=points/np.linalg.norm(points,axis=1,keepdims=True)*100
    expected=[]
    for triangle,p in zip(xyz[faces],points):
        a,b,c=[tuple(row) for row in triangle]
        normal=unit(cross(unit(sub(c,a)),unit(sub(b,a))))
        ratio=dot(normal,a)/dot(normal,p)
        projected=tuple(float(value)*ratio for value in p)
        areas=[]
        for first,second in ((b,c),(a,c),(a,b)):
            vector=cross(sub(first,projected),sub(second,projected))
            areas.append(.5*math.sqrt(dot(vector,vector)))
        total=(areas[0]+areas[1])+areas[2]
        expected.append([value/total for value in areas])
    actual=_triplet_data_weights(torch.tensor(xyz),torch.tensor(faces),torch.arange(3),torch.tensor(points))
    np.testing.assert_array_equal(actual.numpy(),np.array(expected))


class _Samples:
    def sample(self,points,metric):
        return -points[:,0]/100+3


def _two_sample_cost(candidate=None,fold_reference=None):
    xyz,_=_ico(1)
    current=torch.tensor(xyz[:3])
    faces=torch.tensor([[0,1,2]])
    weights=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    source=torch.tensor([1.,2.],dtype=torch.float64)
    layout=_face_layout(faces.numpy(),torch.tensor([0,0]),weights,source,'cpu')
    return _face_costs(current,current if candidate is None else candidate,current,faces,
                       layout,_Samples(),torch.ones(3,dtype=torch.float64),7.5,2,
                       fold_reference=fold_reference)


def test_patch_correlation_with_two_samples_is_used():
    # Official weighted Pearson uses every nonempty varying patch, including
    # two/three samples. The former >=4 guard returned 0.5 here.
    result=_two_sample_cost()
    np.testing.assert_allclose(result,np.zeros((1,8)),atol=1e-14)


def test_fold_cost_replaces_likelihood_and_uses_official_lambda_scale():
    xyz,_=_ico(1)
    candidate=torch.tensor(xyz[:3][[0,2,1]].copy())
    result=_two_sample_cost(candidate)
    assert result[0,7] == 1e7*7.5


def test_sampling_scale_resets_and_returns_unreflected_labels():
    grid,faces=_ico(3)
    centre,samples=_label_samples(grid,faces,15)
    labels=np.vstack((centre,samples))
    scale=1.
    for _ in range(7): _,scale=_rescaled_labels(centre,labels,scale)
    assert scale<.25
    actual,scale=_rescaled_labels(centre,labels,scale)
    np.testing.assert_array_equal(actual,labels)
    assert scale==.8


def test_sphere_origin_and_radius_follow_four_point_estimate():
    xyz,faces=_ico(2)
    translated=xyz+np.array([8.,-3.,5.])
    np.testing.assert_allclose(_normalize_sphere(translated),xyz,atol=5e-13)
    normalized=_normalize_sphere(xyz*1.2)
    np.testing.assert_allclose(np.linalg.norm(normalized,axis=1),100,atol=3e-14)


def test_warp_keeps_coordinate_direction_and_native_order():
    xyz,faces=_ico(2)
    angles=torch.tensor([.03,-.02,.01],dtype=torch.float64)
    rotation=_euler_matrix(angles)
    points=torch.tensor(xyz[[8,2,72,1]])
    moved=_sphere_warp(points,xyz,faces,torch.tensor(xyz)@rotation,'cpu')
    np.testing.assert_allclose(moved.numpy(),(points@rotation).numpy(),atol=6e-13)


def test_rigid_single_feature_pearson_is_spatially_centered_sign():
    xyz,faces=_ico(1)
    source=xyz[:,0]+.31*xyz[:,1]
    cost=_RigidCost(xyz,faces,source,source,'cpu',simval=3)
    np.testing.assert_allclose(cost.centered_source.numpy(),source-source.mean(),atol=0)
    assert cost(torch.eye(3,dtype=torch.float64))>10


def test_unfold_keeps_regular_mesh_unchanged():
    xyz,faces=_ico(2)
    vertices=torch.tensor(xyz)
    result,moved=_unfold(vertices,faces)
    assert moved==0
    assert result is vertices


def test_native_output_qc_checks_actual_float32_gifti_geometry():
    # Solver precision can retain a thin face that becomes degenerate when
    # written. The output count must describe the actual GIFTI coordinates.
    points=np.array([[100.,0.,50.],[100.,1.,50.],[100.,.5,50.+1e-8]])
    faces=np.array([[0,1,2]])
    qc=_native_output_qc(points,faces,points)
    assert qc['folded_solver_faces']==0
    assert qc['folded_output_faces']==1
    assert qc['minimum_solver_orientation_ratio']==1.
    assert qc['minimum_output_orientation_ratio']==0.
    assert qc['degenerate_input_faces']==0


def test_native_output_qc_reports_orientation_flip_without_repairing_coordinates():
    points=np.array([[100.,0.,0.],[100.,1.,0.],[100.,0.,1.]])
    moved=points[[0,2,1]].copy()
    saved=moved.copy()
    qc=_native_output_qc(moved,np.array([[0,1,2]]),points)
    assert qc['folded_solver_faces']==qc['folded_output_faces']==1
    assert qc['minimum_output_orientation_ratio']==-1.
    np.testing.assert_array_equal(moved,saved)


def test_complete_small_mesh_execution_paths_match(tmp_path):
    # Small mathematical fixture checks the full level/fusion/serialization
    # path. Accuracy and timing benchmarks use the server's real sulcal data.
    import nibabel as nib
    from fnit.msm import MSMSulcInputs,run_msmsulc
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        xyz,faces=_ico(2)
        mesh=tmp_path/'sphere.surf.gii';metric=tmp_path/'sulc.shape.gii'
        nib.save(nib.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(xyz.astype(np.float32),intent=1008),
            nib.gifti.GiftiDataArray(faces.astype(np.int32),intent=1009)]),str(mesh))
        values=(xyz[:,0]+.3*xyz[:,1]+.2*xyz[:,2]).astype(np.float32)
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(values,intent=2005)]),str(metric))
        entry=MSMSulcInputs(mesh,mesh,metric,mesh,metric,tmp_path/'unused.mat')
        config=MSMSulcConfig(iterations=(1,1,1,1),control_grid=(1,1,1,1),
                             sampling_grid=(1,2,2,2),data_grid=(1,2,2,2))
        optimized=run_msmsulc({'L':entry,'R':entry},tmp_path/'optimized',device='cpu',
                              config=config,execution='optimized')
        reference=run_msmsulc({'L':entry,'R':entry},tmp_path/'reference',device='cpu',
                              config=config,execution='reference')
        for hemisphere in 'LR':
            np.testing.assert_array_equal(nib.load(str(optimized[hemisphere])).darrays[0].data,
                                           nib.load(str(reference[hemisphere])).darrays[0].data)
    finally:
        torch.set_num_threads(previous)
