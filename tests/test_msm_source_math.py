"""Source-derived MSMSulc formula/schedule regression gates, not benchmarks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from fnit.msm import MSMSulcConfig
from fnit.msm._affine import (
    _RigidCost, _point_matmul, _source_euler_matrix,
    _local_normals, _tangent_basis, _mean_neighbor_distance,
)
from fnit.msm.msmsulc import (
    _ico, _face_costs, _face_layout, _vertex_area,
    _label_samples, _normalize_sphere, _rescaled_labels, _sphere_warp,
    _unfold, _variance_normalize, _native_output_qc, _triplet_data_weights,
    _rotation_matrices, _rotated_label,
)
from fnit.msm._sphere_map import RadialSphereMap


def test_canonical_schedule_has_three_correlation_discrete_levels():
    config=MSMSulcConfig()
    assert config.simval == (3,2,2,2)
    assert config.iterations == (50,10,15,15)
    assert config.control_grid == (6,2,3,4)
    assert config.data_grid == (6,4,5,6)
    assert MSMSulcConfig.ssd_affine().simval == (1,2,2,2)


def test_float_options_match_official_parser_before_double_calculation(tmp_path):
    config=MSMSulcConfig()
    for name,value in (("affine_step_size",.01),("affine_gradient_spacing",.5),
                       ("shear_modulus",.4),("bulk_modulus",1.6),
                       ("strain_exponent",2.),("regularization_exponent",2.)):
        expected=float(np.float32(value))
        assert getattr(config,name)==expected
        assert config.to_dict()[name]==expected
    path=tmp_path/'options.conf'
    path.write_text('--stepsize=.013\n--shearmod=.43\n--lambda=0,10.1,7.6,7.6\n'
                    '--VN\n--rescaleL\n--triclique\n')
    read=MSMSulcConfig.from_file(path)
    assert read.affine_step_size==float(np.float32(.013))
    assert read.shear_modulus==float(np.float32(.43))
    assert read.regularization==tuple(float(np.float32(v)) for v in (0,10.1,7.6,7.6))


@pytest.mark.parametrize('kwargs',[
    {'shear_modulus':1e300},{'regularization':(0,1e300,7.5,7.5)},
    {'affine_step_size':1e-100},
])
def test_float_options_reject_parser_overflow_or_positive_underflow(kwargs):
    with pytest.raises(ValueError,match='float32'):MSMSulcConfig(**kwargs)


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


@pytest.mark.parametrize('option',['numthreads','threads'])
def test_config_accepts_positive_execution_thread_count_only(tmp_path,option):
    path=tmp_path/'config.conf'
    flags='--VN\n--rescaleL\n--triclique\n'
    path.write_text(flags+f'--{option}=8\n')
    assert MSMSulcConfig.from_file(path)==MSMSulcConfig()
    for value in ('0','-2','1.5','invalid'):
        path.write_text(flags+f'--{option}={value}\n')
        with pytest.raises(ValueError):MSMSulcConfig.from_file(path)


def test_variance_normalization_uses_sample_variance():
    actual=_variance_normalize(np.array([0.,2.,5.,8.]))
    np.testing.assert_allclose(actual,(np.array([0.,2.,5.,8.])-3.75)/np.std([0.,2.,5.,8.],ddof=1),rtol=1e-15)
    np.testing.assert_array_equal(_variance_normalize(np.ones(3)),np.zeros(3))


def test_icosphere_retains_constructor_areas_before_midpoint_normalization():
    xyz,faces,cached=_ico(1,cached_area=True)
    # Every planar quarter of the original regular icosahedron face has the
    # same constructor area. Projecting midpoints onto the sphere afterwards
    # makes corner/central faces different; source get_area retains the former.
    tau,one=.8506508084,.5257311121
    triangle=np.array([[-one,0,tau],[one,0,tau],[0,tau,one]])
    cross=np.cross(triangle[2]-triangle[0],triangle[1]-triangle[0])
    expected=.5*np.sqrt((cross[0]**2+cross[1]**2)+cross[2]**2)/4
    # The upstream decimal golden-ratio constants make nominally congruent
    # parent faces differ by up to 3.4e-12 in this quarter-area calculation.
    np.testing.assert_allclose(cached,np.full(len(xyz),expected),rtol=0,atol=4e-12)
    fresh=_vertex_area(xyz/100,faces)
    assert np.ptp(fresh)>1e-3
    assert np.max(np.abs(fresh-cached))>1e-3


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
    rotation=_source_euler_matrix(angles,'cpu')
    points=torch.tensor(xyz[[8,2,72,1]])
    moved=_sphere_warp(points,xyz,faces,torch.tensor(xyz)@rotation,'cpu')
    np.testing.assert_allclose(moved.numpy(),(points@rotation).numpy(),atol=6e-13)


def test_rigid_single_feature_pearson_is_spatially_centered_sign():
    xyz,faces=_ico(1)
    source=xyz[:,0]+.31*xyz[:,1]
    cost=_RigidCost(xyz,faces,source,source,'cpu',simval=3)
    mean=np.add.accumulate(source)[-1]/len(source)
    np.testing.assert_array_equal(cost.centered_source.numpy(),source-mean)
    assert cost.evaluate_positions(cost.vertices)>10



@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('simval',[1,3])
def test_rigid_cost_uses_one_packed_source_wls_boundary(device,simval):
    import math
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    xyz,faces=_ico(1)
    source=xyz[:,0]+.31*xyz[:,1]
    reference=xyz[:,1]-.23*xyz[:,2]
    cost=_RigidCost(xyz,faces,source,reference,device,simval=simval)
    positions=_point_matmul(cost.vertices,_source_euler_matrix((.01,-.02,.03),device))
    native=cost.wls_cost;captured=[]
    def observe(payload,rows,width,sigma):
        assert payload.dtype==np.float64 and payload.flags.c_contiguous
        assert payload.shape==(rows,width,3)
        captured.append(payload.copy())
        return native(payload,rows,width,sigma)
    cost.wls_cost=observe
    actual=cost.evaluate_positions(positions)
    assert len(captured)==1
    payload=captured[0]
    _,_,assigned=cost.mapper.weights(positions)
    ids=cost.ids[assigned].cpu().numpy()
    expected_valid=cost.valid[assigned].cpu().numpy()
    np.testing.assert_array_equal(payload[...,2],expected_valid.astype(np.float64))
    if simval==1:expected_similarity=-np.abs(reference[ids]-source[:,None])
    else:
        mean_source=np.add.accumulate(source)[-1]/len(source)
        mean_reference=np.add.accumulate(reference)[-1]/len(reference)
        expected_similarity=np.sign((reference[ids]-mean_reference)*(source[:,None]-mean_source))
    expected_similarity[ids==0]=0.
    np.testing.assert_array_equal(payload[...,1],expected_similarity)
    # Independent scalar libm/reduction oracle. GPU/CPU geometry may choose
    # different boundary faces; each uses the same source WLS definition.
    total=0.
    for row in payload:
        weight_sum=value_sum=0.
        for distance,similarity,valid in row:
            if valid and distance>0:
                weight=math.exp(-float(distance)/((2*cost.sigma)*cost.sigma))
                weight_sum+=weight;value_sum+=float(similarity)*weight
        total+=value_sum/weight_sum if weight_sum else value_sum
    assert np.float64(actual).view(np.uint64)==np.float64(total).view(np.uint64)


def test_rigid_local_normals_and_tangent_basis_follow_scalar_source_order():
    import math
    def cross(a,b):return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
    def dot(a,b):return (a[0]*b[0]+a[1]*b[1])+a[2]*b[2]
    def normalize(a):
        norm=math.sqrt(dot(a,a))
        return a/norm if norm>1e-8 else a
    xyz,faces=_ico(1)
    normal=[];first=[];second=[]
    for vertex in range(len(xyz)):
        total=np.zeros(3)
        for triangle in faces:
            if vertex in triangle:
                a,b,c=xyz[triangle]
                total+=normalize(cross(c-a,b-a))
        n=normalize(total)
        if dot(n,xyz[vertex])<0:n=-n
        normal.append(n)
        x,y,z=n
        if abs(x)>=abs(y) and abs(x)>=abs(z):
            magnitude=math.sqrt(z*z+y*y)
            e1=np.array([0,-z/magnitude,y/magnitude]) if magnitude else np.array([0,0,1.])
        elif abs(y)>=abs(x) and abs(y)>=abs(z):
            magnitude=math.sqrt(z*z+x*x)
            e1=np.array([-z/magnitude,0,x/magnitude]) if magnitude else np.array([0,0,1.])
        else:
            magnitude=math.sqrt(y*y+x*x)
            e1=np.array([-y/magnitude,x/magnitude,0]) if magnitude else np.array([1.,0,0])
        first.append(e1);second.append(normalize(cross(n,e1)))
    actual=_local_normals(torch.tensor(xyz),torch.tensor(faces))
    actual_first,actual_second=_tangent_basis(actual)
    np.testing.assert_array_equal(actual.numpy(),np.asarray(normal))
    np.testing.assert_array_equal(actual_first.numpy(),np.asarray(first))
    np.testing.assert_array_equal(actual_second.numpy(),np.asarray(second))


def test_rigid_mean_distance_keeps_directed_edge_insertion_order():
    import math
    xyz,faces=_ico(1)
    neighbors=[[] for _ in xyz]
    for a,b,c in faces:
        for vertex,others in ((a,(b,c)),(b,(a,c)),(c,(a,b))):
            for other in others:
                if other not in neighbors[vertex]:neighbors[vertex].append(other)
    total=0.;count=0
    for vertex,items in enumerate(neighbors):
        for other in items:
            x,y,z=xyz[other]-xyz[vertex]
            total+=math.sqrt((x*x+y*y)+z*z);count+=1
    assert _mean_neighbor_distance(xyz,faces)==total/count


def test_affine_point_product_preserves_source_three_term_order():
    points=torch.tensor([[1e16,-1e16,1.]],dtype=torch.float64)
    matrix=torch.ones((3,3),dtype=torch.float64)
    torch.testing.assert_close(_point_matmul(points,matrix),torch.ones((1,3),dtype=torch.float64),rtol=0,atol=0)
    angles=np.array([.013,-.02,.04])
    a,b,c=angles;ca,cb,cc=np.cos(angles);sa,sb,sc=np.sin(angles)
    rx=np.array([[1.,0.,0.],[0.,ca,-sa],[0.,sa,ca]])
    ry=np.array([[cb,0.,sb],[0.,1.,0.],[-sb,0.,cb]])
    rz=np.array([[cc,-sc,0.],[sc,cc,0.],[0.,0.,1.]])
    np.testing.assert_allclose(_source_euler_matrix(angles,'cpu').numpy(),
                               rz@ry@rx,rtol=0,atol=2e-16)



@pytest.mark.parametrize('device',['cpu','cuda'])
def test_cached_label_rotation_preserves_source_three_term_order(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    rotations=torch.ones((2,3,3),dtype=torch.float64,device=device)
    actual=_rotated_label(rotations,np.array([1e16,-1e16,1.]))
    torch.testing.assert_close(actual,torch.ones((2,3),dtype=torch.float64,device=device),rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_source_rotation_cache_matches_native_matrix_buffer(device):
    from fnit.msm import _fastpd_native
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    xyz,_=_ico(1)
    centre=xyz[12]
    expected=np.frombuffer(_fastpd_native.source_rotation_matrices(xyz,centre,len(xyz)),np.float64).reshape(-1,3,3)
    cached=_rotation_matrices(xyz,centre,device)
    np.testing.assert_array_equal(cached.cpu().numpy(),expected)
    # The zero label is still transformed, rather than replacing the source
    # operation with the prior CP coordinates.
    label,_=_rescaled_labels(centre,centre[None,:],1.)
    actual=_rotated_label(cached,label[0]).cpu().numpy()
    scalar=[]
    for matrix in expected:
        scalar.append([(matrix[i,0]*label[0,0]+matrix[i,1]*label[0,1])+matrix[i,2]*label[0,2]
                       for i in range(3)])
    np.testing.assert_array_equal(actual,np.asarray(scalar))


def test_affine_returns_statefully_rotated_mesh_not_one_accumulated_product(monkeypatch):
    from fnit.msm import _affine as implementation
    class KnownCost:
        def __init__(self,vertices,*args):self.vertices=torch.tensor(vertices,dtype=torch.float64)
        def evaluate_positions(self,positions):
            return float((positions[:,0]+.3*positions[:,1]-.2*positions[:,2]).sum())
    monkeypatch.setattr(implementation,'_RigidCost',KnownCost)
    xyz=np.array([[18.,2.,80.],[-31.,70.,3.],[9.,17.,43.]])
    result=implementation._affine_initialization(xyz,np.empty((0,3)),None,None,'cpu',
             MSMSulcConfig(iterations=(2,1,1,1)))
    matrix,_,_,positions=result
    # Source keeps the coordinate rounding of every accepted rotation.
    assert np.any(positions.numpy()!=xyz@matrix)
    np.testing.assert_allclose(positions.numpy(),xyz@matrix,atol=3e-14,rtol=0)


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


def test_complete_small_mesh_execution_paths_match(tmp_path,monkeypatch):
    # Small mathematical fixture checks the full level/fusion/serialization
    # path. Accuracy and timing benchmarks use the server's real sulcal data.
    import nibabel as nib
    from fnit.msm import MSMSulcInputs,run_msmsulc,_fastpd_native
    # Source get_rotations runs once per iteration, then all fusion labels
    # reuse those matrices. Keep that boundary independent of label count.
    rotation_calls=[]
    original_rotation=_fastpd_native.source_rotation_matrices
    def observe_rotation(prior,centre,count):
        rotation_calls.append(count)
        return original_rotation(prior,centre,count)
    monkeypatch.setattr(_fastpd_native,'source_rotation_matrices',observe_rotation)
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
        assert rotation_calls==[42]*12  # 3 stages × 2 hemispheres × 2 executions
        for hemisphere in 'LR':
            np.testing.assert_array_equal(nib.load(str(optimized[hemisphere])).darrays[0].data,
                                           nib.load(str(reference[hemisphere])).darrays[0].data)
    finally:
        torch.set_num_threads(previous)


def test_final_native_output_preserves_source_transform_and_reports_fold(tmp_path,monkeypatch):
    # Mathematical workflow gate, not a real-data benchmark: a controlled
    # terminal native deformation verifies direct source-compatible output.
    import json
    import nibabel as nib
    from fnit.msm import MSMSulcInputs,run_msmsulc
    from fnit.msm import msmsulc as implementation
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        xyz,faces=_ico(2)
        mesh=tmp_path/'sphere.surf.gii';metric=tmp_path/'sulc.shape.gii'
        nib.save(nib.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(xyz.astype(np.float32),intent=1008),
            nib.gifti.GiftiDataArray(faces.astype(np.int32),intent=1009)]),str(mesh))
        nib.save(nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(
            (xyz[:,0]+.3*xyz[:,1]).astype(np.float32),intent=2005)]),str(metric))
        entry=MSMSulcInputs(mesh,mesh,metric,mesh,metric,tmp_path/'unused.mat')
        config=MSMSulcConfig(iterations=(1,1,1,1),control_grid=(1,1,1,1),
                             sampling_grid=(1,2,2,2),data_grid=(1,1,1,1))
        original_warp=implementation._sphere_warp
        native_calls=0;expected=[]
        def terminal_warp(points,*args,**kwargs):
            nonlocal native_calls
            result=original_warp(points,*args,**kwargs)
            if len(points)==len(xyz):
                native_calls+=1
                if native_calls%4==0:
                    result=result.clone()
                    first,second=int(faces[0,1]),int(faces[0,2])
                    result[[first,second]]=result[[second,first]]
                    expected.append(result.detach().cpu().numpy().astype(np.float32))
            return result
        monkeypatch.setattr(implementation,'_sphere_warp',terminal_warp)
        outputs=run_msmsulc({'L':entry,'R':entry},tmp_path/'output',device='cpu',config=config)
        report=json.loads((tmp_path/'output/registration_report.json').read_text())
        assert len(expected)==2
        for i,hemisphere in enumerate('LR'):
            written=nib.load(str(outputs[hemisphere])).darrays[0].data
            np.testing.assert_array_equal(written,expected[i])
            assert report[hemisphere]['folded_output_faces']>0
            assert report[hemisphere]['folded_solver_faces']>0
    finally:
        torch.set_num_threads(previous)
