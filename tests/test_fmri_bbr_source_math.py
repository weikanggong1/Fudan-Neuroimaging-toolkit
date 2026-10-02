"""Offline NEWIMAGE BBR geometry and cost oracles, not a real-data benchmark."""

import itertools

import nibabel as nib
import numpy as np
import pytest
import torch

from fnit.fmri.bbr import _BBRCost, _boundary, _sample_zero, _smooth_wm


def _volume(data, spacing=(1., 1., 1.), neurological=False):
    affine = np.diag((*spacing, 1.))
    if not neurological:
        affine[0, 0] *= -1
    return nib.Nifti1Image(np.asarray(data, np.float32), affine)


def _smooth_oracle(data, spacing):
    result = np.asarray(data, np.float32)
    for axis, size in enumerate(spacing):
        sigma = np.float32(2. / np.float32(size))
        radius = int(float(sigma) - .001) * 2 + 3
        kernel = np.array([np.float32(np.exp(-j*j/(2.*float(sigma)**2)))
                           for j in range(-radius, radius + 1)], np.float64)
        total = np.float32(0)
        for value in kernel:
            total = np.float32(total + np.float32(value))
        kernel /= float(total)
        pads = [(0, 0)] * 3
        pads[axis] = (radius, radius)
        padded = np.pad(result, pads)
        output = np.zeros_like(result)
        for index, coefficient in enumerate(kernel):
            slices = [slice(None)] * 3
            slices[axis] = slice(index, index + result.shape[axis])
            output = (output.astype(np.float64)
                      + padded[tuple(slices)].astype(np.float64) * coefficient).astype(np.float32)
        result = output
    return result


@pytest.mark.parametrize('spacing', [(1.,1.,1.), (1.2,1.5,2.1)])
def test_smoothing_matches_source_kernel_radius_and_per_tap_float_rounding(spacing):
    data = np.zeros((9, 10, 11), np.float32)
    data[0:6, 3:8, 2:7] = 1.
    actual = _smooth_wm(torch.from_numpy(data), spacing).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32), _smooth_oracle(data, spacing).view(np.uint32))


@pytest.mark.parametrize('neurological', [False, True])
def test_boundary_matches_source_27_tap_gradient_and_x_fast_order(neurological):
    data = np.zeros((12, 13, 14), np.float32)
    data[1:9, 2:11, 3:12] = 1.
    image = _volume(data, neurological=neurological)
    gm, wm, _ = _boundary(image, image)
    source = data[::-1].copy() if neurological else data
    smooth = _smooth_oracle(source, (1.,1.,1.))
    padded = np.pad(source, 1)
    neighbours = np.zeros_like(source)
    for x,y,z in itertools.product(range(3), repeat=3):
        neighbours += padded[x:x+12, y:y+13, z:z+14]
    points = np.argwhere(((source > .5) & (neighbours < 26.5)).transpose(2,1,0))[:, ::-1]
    padded = np.pad(smooth, 1)
    gradient = np.zeros((len(points), 3), np.float32)
    for z,y,x in itertools.product((-1,0,1), repeat=3):
        values = padded[points[:,0]+x+1, points[:,1]+y+1, points[:,2]+z+1]
        for axis,(a,b,c) in enumerate(((x,y,z),(y,x,z),(z,x,y))):
            gradient[:,axis] += values * np.float32(a * 3.**(1-abs(b)-abs(c)))
    normal = -gradient
    norm = np.sqrt((normal[:,0]**2 + normal[:,1]**2) + normal[:,2]**2)
    normal /= norm[:,None]
    np.testing.assert_array_equal(gm, (points + (2.*normal).astype(np.float64)).astype(np.float32))
    np.testing.assert_array_equal(wm, (points - (2.*normal).astype(np.float64)).astype(np.float32))


def test_sampling_uses_zero_corners_instead_of_infinite_border_replication():
    image = torch.full((3,4,5), 2., dtype=torch.float32)
    coords = torch.tensor([[-.25, 1.,2.],[-1.25,1.,2.],[2.25,1.,2.],[3.25,1.,2.]])
    torch.testing.assert_close(_sample_zero(image, coords), torch.tensor([1.5,0.,1.5,0.]), rtol=0, atol=0)


def test_cost_uses_double_contrast_and_returns_float32():
    data = np.indices((8,9,10), dtype=np.float32)
    image = _volume(20. + 3.*data[0] + data[1] + .3*data[2])
    gm = np.array([[3.,3.,3.],[4.,3.,3.],[5.,4.,3.]], np.float32)
    wm = gm - np.array([1.,0.,0.], np.float32)
    cost = _BBRCost(image, gm, wm, 'cpu')
    samples = _sample_zero(cost.image, torch.from_numpy(np.stack((gm,wm)))).numpy().astype(np.float64)
    contrast = 200.*(samples[0]-samples[1])/(samples[0]+samples[1])
    expected = np.float32(np.mean(1.+np.tanh(-.5*contrast)))
    actual = cost.evaluate(np.eye(4), step=1)
    assert actual.dtype == torch.float32
    np.testing.assert_array_equal(actual.numpy(), np.array([expected], np.float32))


@pytest.mark.parametrize('batch_size', [1,3,128])
def test_batched_cost_matches_scalar_and_chunking(batch_size):
    data = np.indices((8,9,10), dtype=np.float32)
    image = _volume(20. + 3.*data[0] + data[1] + .3*data[2])
    gm = np.array([[3.,3.,3.],[4.,3.,3.],[5.,4.,3.]], np.float32)
    wm = gm - np.array([1.,0.,0.], np.float32)
    cost = _BBRCost(image, gm, wm, 'cpu', max_batch_size=batch_size)
    matrices = np.repeat(np.eye(4)[None], 7, axis=0)
    matrices[:,0,3] = np.arange(7)*.13
    actual = cost.evaluate(matrices, step=1).numpy()
    reference = np.array([cost(matrix, step=1) for matrix in matrices], np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
    assert cost.evaluations == 14


def test_sampled_boundary_below_100_points_is_not_rejected():
    data = np.zeros((6,6,6), np.float32)
    data[2:4,2:4,2:4] = 1
    image = _volume(data)
    gm, wm, _ = _boundary(image,image)
    assert gm.shape == wm.shape == (8,3)


def test_grid_uses_float_parsed_schedule_values_and_source_odometer_order():
    from fnit.fmri.bbr import _grid_perturbations
    rotation, translation = float(np.float32(.07)), float(np.float32(4.))
    values = _grid_perturbations(.07,4.)
    assert values.shape == (729,12)
    expected = np.zeros((729,12))
    pointer = np.zeros(6, dtype=np.int64)
    for index in range(729):
        carry = 1
        for axis in range(6):
            pointer[axis] += carry
            if pointer[axis] < 3:
                carry = 0
            else:
                pointer[axis] = 0
        expected[index,:6] = (pointer-1)*np.array([rotation]*3+[translation]*3)
    np.testing.assert_array_equal(values,expected)
    np.testing.assert_array_equal(values[0,:6], [0.,-rotation,-rotation,-4.,-4.,-4.])
    np.testing.assert_array_equal(values[-1,:6], [-rotation]*3+[-4.]*3)
    assert np.count_nonzero((values==0).all(1)) == 1


def test_powell_matches_unmodified_miscmath_source_oracle_with_direction_update():
    # Generated offline from vendored optimise.cc, linked only to a minimal
    # double-vector adapter. It performed one genuine Powell direction update.
    from fnit.fmri.bbr import _powell_optimize
    point = np.array([-.7,.6,.1,.2,.3,.4,1.,1.,1.,0.,0.,0.])
    tolerance = np.array([.0005]*3+[.02]*3+[.002]*3+[.001]*3,np.float32).astype(np.float64)
    evaluations = []
    def objective(p):
        evaluations.append(p.copy())
        u=p[0]+.7*p[1]-.35
        v=.6*p[0]-p[1]+.2
        c=u*u+2*v*v+.04*(p[0]-.5)**4
        for i in range(2,6):
            c += .1*(p[i]-.07*(i+1))**2
        return float(np.float32(c))
    result,cost = _powell_optimize(point,tolerance,objective)
    expected = np.array([.15000498674854754,.2891576267064489,.20999975353479386,
                         .2799998864531517,.3499999038875103,.4199999995529652,
                         1.,1.,1.,0.,0.,0.])
    np.testing.assert_allclose(result,expected,atol=1e-12,rtol=0)
    assert cost == .0006074788980185986
    assert len(evaluations) == 131


def test_boundary_force_one_reference_sampling_is_separate_from_wm_spacing():
    data = np.zeros((12,13,14),np.float32)
    data[2:9,2:10,3:11] = 1
    image = _volume(data,spacing=(1.,1.5,2.))
    gm_native,wm_native,_ = _boundary(image,image)
    gm_force,wm_force,_ = _boundary(image,image,reference_sampling=(1.,1.,1.))
    np.testing.assert_allclose((gm_native+wm_native)/2, (gm_force+wm_force)/2,atol=1e-6)
    assert np.max(abs(gm_force-gm_native)) > .1


def test_boundary_rejects_nonfinite_segmentation():
    data = np.zeros((6,6,6),np.float32)
    data[2:4,2:4,2:4] = 1
    data[0,0,0] = np.nan
    image = _volume(data)
    with pytest.raises(ValueError,match='finite'):
        _boundary(image,image)


def test_bbr_exposes_stage_timings_and_reference_candidate_path():
    from fnit.fmri.bbr import register_bbr
    grid=np.indices((16,16,16),dtype=np.float32)
    radius=np.sqrt(sum((grid[k]-8)**2 for k in range(3)))
    wm=_volume((radius<4).astype(np.float32))
    epi=_volume(np.where(radius<4,80.,np.where(radius<6,100.,0.)).astype(np.float32))
    result=register_bbr(epi,epi,wm,init=np.eye(4),device='cpu',grid_search=False,execution='reference')
    assert set(result.phase_timings) == {'initial_flirt','boundary_preparation','coarse_bbr','local_bbr','final_resampling'}
    assert result.cost_evaluations > 0
    assert set(result.phase_cost_evaluations) == {'coarse_bbr','local_bbr'}
    assert result.host_result_transfers == result.cost_evaluations


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('neurological',[False,True])
def test_cuda_boundary_matches_cpu_float_bits_with_tf32_enabled(neurological):
    data=np.zeros((12,13,14),np.float32)
    data[1:9,2:11,3:12]=1
    image=_volume(data,neurological=neurological)
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=True
        expected=_boundary(image,image,device='cpu')
        actual=_boundary(image,image,device='cuda:0')
        for before,after in zip(expected[:2],actual[:2]):
            np.testing.assert_array_equal(before.view(np.uint32),after.view(np.uint32))
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32=old


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('batch_size',[1,3,128])
def test_cuda_cost_matches_cpu_and_scalar_bits(batch_size):
    rng=np.random.default_rng(9)
    image=_volume(rng.uniform(40,100,(15,16,17)).astype(np.float32))
    gm=rng.uniform(-1,16,(37,3)).astype(np.float32)
    wm=gm-np.array([1.,0.,0.],np.float32)
    matrices=np.repeat(np.eye(4)[None],9,axis=0)
    matrices[:,0,3]=np.linspace(-.8,.8,9)
    expected=_BBRCost(image,gm,wm,'cpu').evaluate(matrices,step=1).numpy()
    cost=_BBRCost(image,gm,wm,'cuda:0',max_batch_size=batch_size)
    actual=cost.evaluate(matrices,step=1).cpu().numpy()
    scalar=np.array([cost(m,step=1) for m in matrices],np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(actual.view(np.uint32),scalar.view(np.uint32))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('scale',[1.e-8,1.,1.e8])
def test_fused_cost_matches_tensor_near_denominator_and_fov_boundaries(scale):
    grid=np.indices((6,7,8),dtype=np.float32)
    image=_volume(scale*(.1+grid[0]+.07*grid[1]+.01*grid[2]))
    x=np.array([-1.25,-1.,-.25,np.nextafter(np.float32(0),np.float32(-np.inf)),0.,.25,5.,5.25,6.,6.25],np.float32)
    gm=np.stack((x,np.full_like(x,3.),np.full_like(x,4.)),1)
    wm=gm-np.array([.5,0.,0.],np.float32)
    matrices=np.repeat(np.eye(4)[None],3,axis=0)
    matrices[1,0,3]=.2
    matrices[2,1,3]=-.1
    fused=_BBRCost(image,gm,wm,'cuda:0',use_fused=True).evaluate(matrices,step=1)
    tensor=_BBRCost(image,gm,wm,'cuda:0',use_fused=False).evaluate(matrices,step=1)
    torch.testing.assert_close(fused,tensor,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_fused_and_tensor_complete_schedule_return_same_matrix_and_image():
    from fnit.fmri.bbr import register_bbr
    grid=np.indices((24,26,28),dtype=np.float32)
    radius=np.sqrt(sum((grid[k]-(12,13,14)[k])**2 for k in range(3)))
    wm=_volume((radius<6).astype(np.float32))
    epi=_volume(np.where(radius<6,80.,np.where(radius<9,100.,0.)).astype(np.float32))
    initial=np.eye(4);initial[0,3]=.7
    reference=register_bbr(epi,epi,wm,init=initial,device='cuda:0',grid_search=True,execution='reference')
    fused=register_bbr(epi,epi,wm,init=initial,device='cuda:0',grid_search=True,execution='batched')
    np.testing.assert_array_equal(reference.matrix,fused.matrix)
    np.testing.assert_array_equal(np.asarray(reference.moved.dataobj),np.asarray(fused.moved.dataobj))
    assert reference.final_cost == fused.final_cost
    assert reference.cost_evaluations == fused.cost_evaluations
    assert fused.host_result_transfers < reference.host_result_transfers


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('neurological',[False,True])
def test_nifti_proxy_layout_matches_tensor_cost_and_complete_fused_schedule(tmp_path,neurological):
    from fnit.fmri.bbr import register_bbr
    rng=np.random.default_rng(19)
    data=rng.uniform(40,100,(17,18,19)).astype(np.float32)
    path=tmp_path/'random.nii.gz'
    nib.save(_volume(data,neurological=neurological),path)
    loaded=nib.load(path)
    assert np.asarray(loaded.dataobj).flags.f_contiguous  # Actual NIfTI ArrayProxy storage.
    gm=rng.uniform(2,14,(131,3)).astype(np.float32)
    wm=gm-np.array([.8,0.,0.],np.float32)
    matrices=np.repeat(np.eye(4)[None],7,axis=0)
    matrices[:,0,3]=np.linspace(-.7,.7,7)
    tensor=_BBRCost(loaded,gm,wm,'cuda:0',use_fused=False)
    fused=_BBRCost(loaded,gm,wm,'cuda:0',use_fused=True)
    torch.testing.assert_close(fused.evaluate(matrices,step=1),tensor.evaluate(matrices,step=1),rtol=0,atol=0)
    assert fused.image.is_contiguous()
    grid=np.indices((24,26,28),dtype=np.float32)
    radius=np.sqrt(sum((grid[k]-(12,13,14)[k])**2 for k in range(3)))
    wm_image=_volume((radius<6).astype(np.float32),neurological=neurological)
    epi_image=_volume(np.where(radius<6,80.,np.where(radius<9,100.,0.)).astype(np.float32),neurological=neurological)
    epi_path,wm_path=tmp_path/'epi.nii.gz',tmp_path/'wm.nii.gz'
    nib.save(epi_image,epi_path);nib.save(wm_image,wm_path)
    initial=np.eye(4);initial[0,3]=.7  # Optimize from initialization, not the solution.
    reference=register_bbr(epi_path,epi_path,wm_path,init=initial,device='cuda:0',grid_search=True,execution='reference')
    actual=register_bbr(epi_path,epi_path,wm_path,init=initial,device='cuda:0',grid_search=True,execution='batched')
    np.testing.assert_array_equal(actual.matrix,reference.matrix)
    np.testing.assert_array_equal(np.asarray(actual.moved.dataobj),np.asarray(reference.moved.dataobj))
    assert actual.final_cost == reference.final_cost
    assert actual.cost_evaluations == reference.cost_evaluations
