#!/usr/bin/env python

import contextlib
import tempfile
import os
import shutil
import shlex
import os.path as op

import pytest

import numpy               as np
import nibabel.streamlines as nibtrk

from fsl.data.image  import Image
from fsl_streamlines import main


thisdir = op.dirname(op.abspath(__file__))
datadir = op.join(thisdir, 'testdata')


@contextlib.contextmanager
def tempdir():
    with tempfile.TemporaryDirectory() as td:
        prevdir = os.getcwd()

        os.chdir(td)
        try:
            yield td
        finally:
            os.chdir(prevdir)


def compare_tractograms(trk1, trk2):
    assert np.all(np.isclose(trk1.streamlines.get_data(),
                             trk2.streamlines.get_data()))

    assert (trk1.tractogram.data_per_streamline.keys() ==
            trk2.tractogram.data_per_streamline.keys())
    assert (trk1.tractogram.data_per_point.keys() ==
            trk2.tractogram.data_per_point.keys())

    for key in trk1.tractogram.data_per_streamline.keys():
        d1 = trk1.tractogram.data_per_streamline[key].get_data()
        d2 = trk2.tractogram.data_per_streamline[key].get_data()
        assert np.all(np.isclose(d1, d2))

    for key in trk1.tractogram.data_per_point.keys():
        d1 = trk1.tractogram.data_per_point[key].get_data()
        d2 = trk2.tractogram.data_per_point[key].get_data()
        assert np.all(np.isclose(d1, d2))


def test_help():
    with pytest.raises(SystemExit) as e:
        main.main(['-h'])
        assert e.code == 0


def test_version():
    with pytest.raises(SystemExit) as e:
        main.main(['-v'])
        assert e.code == 0


@pytest.mark.parametrize('njobs', [1, 8])
def test_xtract_input(njobs):
    with tempdir():
        tract_dir = 'xtract/tracts/cst_l/'
        os.makedirs(tract_dir)
        shutil.copy(f'{datadir}/densityNorm.nii.gz',      tract_dir)
        shutil.copy(f'{datadir}/density_localdir.nii.gz', tract_dir)

        cmd = f'{tract_dir} -nj {njobs} -ss 40 -t 0.01'
        main.main(shlex.split(cmd))

        assert op.exists(f'{tract_dir}/streamlines.trk')


@pytest.mark.parametrize('njobs', [1, 8])
def test_ptx2_input(njobs):
    with tempdir():
        tract_dir = 'ptx2/'
        os.makedirs(tract_dir)
        shutil.copy(f'{datadir}/waytotal',                f'{tract_dir}/waytotal')
        shutil.copy(f'{datadir}/density.nii.gz',          f'{tract_dir}/fdt_paths.nii.gz')
        shutil.copy(f'{datadir}/density_localdir.nii.gz', f'{tract_dir}/fdt_paths_localdir.nii.gz')

        cmd = f'{tract_dir} -nj {njobs} -ss 40 -t 0.01'
        main.main(shlex.split(cmd))

        assert op.exists(f'{tract_dir}/streamlines.trk')


@pytest.mark.parametrize('njobs', [1, 8])
def test_single_file_input(njobs):
    with tempdir():
        tract_file = 'orientation.nii.gz'
        shutil.copy(f'{datadir}/density_localdir.nii.gz', tract_file)

        cmd = f'{tract_file} -nj {njobs} -ss 40'
        main.main(shlex.split(cmd))

        assert op.exists('streamlines.trk')


@pytest.mark.parametrize('njobs', [1, 8])
def test_trx_output(njobs):
    with tempdir():
        tract_file = 'orientation.nii.gz'
        shutil.copy(f'{datadir}/density_localdir.nii.gz', tract_file)

        cmd = f'{tract_file} -nj {njobs} -ss 40 -f trx'
        main.main(shlex.split(cmd))

        assert op.exists('streamlines.trx')


@pytest.mark.parametrize('njobs', [1, 8])
def test_jitter_seeds_per_voxel(njobs):
    with tempdir():
        tract_file = 'orientation.nii.gz'
        shutil.copy(f'{datadir}/density_localdir.nii.gz', tract_file)

        cmd = f'{tract_file} -nj {njobs} -ss 40 -j -spv 5'
        main.main(shlex.split(cmd))

        assert op.exists('streamlines.trk')



def test_zero_orientation():
    with tempdir():
        tract_dir    = 'xtract/tracts/cst_l/'
        orient_file  = f'{tract_dir}/orientations.nii.gz'
        mask_file    = f'{tract_dir}/seed_mask.nii.gz'
        os.makedirs(tract_dir)

        orient = Image(f'{datadir}/density_localdir.nii.gz')
        mask   = Image(np.zeros(orient.shape[:3], dtype=np.uint8),
                       header=orient.header)

        orient[100, 100, 100] = [0, 0, 0]
        mask[  100, 100, 100] = 1

        orient.save(orient_file)
        mask  .save(mask_file)

        cmd = f'{orient_file} -m {mask_file}'
        main.main(shlex.split(cmd))

        assert not op.exists(f'{tract_dir}/streamlines.trk')


def test_seed_mask():
    with tempdir():
        tract_dir    = 'xtract/tracts/cst_l/'
        orient_file  = f'{tract_dir}/density_localdir.nii.gz'
        density_file = f'{tract_dir}/densityNorm.nii.gz'
        mask_file    = f'{tract_dir}/seed_mask.nii.gz'
        os.makedirs(tract_dir)

        shutil.copy(f'{datadir}/densityNorm.nii.gz',      density_file)
        shutil.copy(f'{datadir}/density_localdir.nii.gz', orient_file)

        # create a mask from the density image
        thr     = 0.05
        density = Image(density_file)
        mask    = Image(density)
        mask[:] = density.data > thr
        mask.save(mask_file)

        # compare with using density at same threshold.
        cmd1 = f'{tract_dir} -ss 20 -o from_density -t {thr}'

        # We still need to provide the same threshold
        # as it is used during tracking to truncate
        # streamlines
        cmd2 = f'{tract_dir} -ss 20 -o from_mask -t {thr} -m {mask_file}'

        main.main(shlex.split(cmd1))
        main.main(shlex.split(cmd2))

        assert op.exists(f'{tract_dir}/from_density.trk')
        assert op.exists(f'{tract_dir}/from_mask.trk')

        trk1 = nibtrk.load(f'{tract_dir}/from_density.trk')
        trk2 = nibtrk.load(f'{tract_dir}/from_mask.trk')

        compare_tractograms(trk1, trk2)


def test_vtk_save_overlay():
    with tempdir():
        tract_file = 'orientation.nii.gz'
        shutil.copy(f'{datadir}/density_localdir.nii.gz', tract_file)
        shutil.copy(f'{datadir}/density.nii.gz',          '.')

        cmd = f'{tract_file} -ss 40 -f vtk -so density'
        main.main(shlex.split(cmd))

        assert op.exists('streamlines.vtk')
        assert op.exists('density.vti')
