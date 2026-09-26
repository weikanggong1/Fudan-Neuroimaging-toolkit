"""
fsl-streamlines: Streamlining class
"""

import itertools as it
import os
import sys
import multiprocessing as mp

from scipy import ndimage
import numpy as np

from fsl.data.image import Image, removeExt

from .util import write_image_as_vtk


class NoSeedsError(Exception):
    """Exception raised by the Streamliner if there are no
    voxels above the density threshold.
    """


class Streamliner:
    """
    Class to generate streamlines from output of probtrackx2 / xtract
    """

    def __init__(self,
                 orientation,
                 density=None,
                 threshold=1e-3,
                 seed_mask=None,
                 nsteps=300,
                 stepsize=.4,
                 jitter=False,
                 n_per_vox=1,
                 min_n_steps=-1,
                 subsample=1,
                 order=0,
                 num_jobs=1):
        """
        Generate streamlines

        :orientation: FSLpy Image object containing streamline orientation
        :density: FSLpy Image object containing streamline density (may be None)
        :threshold: Density threshold to include a voxel as a seed
        :seed_mask: Binary mask defining seeds
        :nsteps: Maximum number of steps to take for a streamline
        :stepsize: Step size in mm
        :jitter: If True, seed randomly from within a voxel. Useful when n_per_vox > 1
        :n_per_vox: Streamlines per seed voxel - forced to equal 1 if jitter=False
        :min_n_steps: Minimum number of steps for streamline to be included. If < 0, no minimum
        :subsample: Subsample seed positions by this factor - useful for fast rough output
        :order: Density interpolation order (0=nearest, 1=linear, 3=cubic)
        :num_jobs: Number of jobs to use to parallelise streamlining
        """

        self.orientation = orientation.data
        self.shape = orientation.shape[:3]

        if density is not None:
            self.density = density.data
        else:
            self.density = None

        self.affine = orientation.getAffine('voxel', 'world')
        self.vox_size = np.array(orientation.pixdim[:3])
        # tracking params
        self.threshold = threshold
        self.nsteps = nsteps
        self.stepsize = stepsize
        self.stepvox = self.stepsize / self.vox_size
        self.jitter = jitter
        self.subsample = subsample
        self.n_per_vox = n_per_vox if jitter else 1
        self.min_n_steps = min_n_steps

        # seed from mask
        if seed_mask is not None:
            seeds = np.argwhere(seed_mask.data > 0)

        # Seed from above-threshold density voxels
        elif density is not None:
            seeds = np.argwhere(density.data > threshold)

        # Seed from non-zero orientation voxels
        else:
            nzmask = ((self.orientation[..., 0] != 0) |
                      (self.orientation[..., 1] != 0) |
                      (self.orientation[..., 2] != 0))
            seeds = np.argwhere(nzmask)

        self.seeds = seeds[::subsample, :]
        self.order = order
        self.num_jobs = num_jobs

        if len(self.seeds) == 0:
            raise NoSeedsError(f"No seed voxels with density above threshold {threshold}!")

        # pre-filtered density image for spline
        # interpolation of density values
        if density is not None:
            if order > 1:
                self.filtered_density = ndimage.spline_filter(self.density, self.order)
            else:
                self.filtered_density = self.density
        else:
            self.filtered_density = None


    def _pos_to_voxel(self, pos):
        # Round coordinates to voxels
        return np.round(pos).astype(int)


    def _check_position(self, pos):
        # check that position is in bounds and above density threshold
        ix, iy, iz = self._pos_to_voxel(pos)

        good = (0 <= ix < self.shape[0] and
                0 <= iy < self.shape[1] and
                0 <= iz < self.shape[2])
        if self.density is None:
            return good
        else:
            return good and (self.density[ix, iy, iz] >= self.threshold)


    def _do_jitter(self, pos):
        if not self.jitter:
            return pos
        else:
            return pos + np.random.uniform(-.5, .5, 3)


    @staticmethod
    def _merge_streamlines(st1, st2):

        # Keep streamlines where only forward- or
        # backwards-tracking produced something
        if  (st1 is None) and (st2 is None): return None
        elif st1 is None:                    return st2
        elif st2 is None:                    return st1

        # Merge both together, flipping one of them
        return np.concatenate((np.flipud(st1), st2[1:, :]), axis=0)


    def streamline(self):
        # The main loop - streamline from all seed voxels
        print(f'Streamlining from {len(self.seeds)} voxels...')

        # Divide seeds into blocks, with each block
        # handled by a separate process. This runs
        # a bit faster than just using pool.map on
        # each seed separately
        if self.num_jobs == 1:
            result = [self._streamline_from_seeds(self.seeds)]
        else:
            nseeds        = len(self.seeds)
            seeds_per_job = np.ceil(nseeds / self.num_jobs).astype(int)
            starts        = range(0, nseeds, seeds_per_job)
            seed_blocks   = [self.seeds[s:s+seeds_per_job] for s in starts]

            with mp.Pool(self.num_jobs) as pool:
                result = pool.map(self._streamline_from_seeds, seed_blocks)

        all_streamlines = []
        all_densities = []

        # The _streamline_from_seeds function returns
        # all streamlines in a single Nx3 array, along
        # with offsets into that array for each streamline
        for st, offs, de in result:
            if st is None:
                continue
            all_streamlines.extend(np.array_split(st, offs))
            if de is not None:
                all_densities.extend(np.array_split(de, offs))

        if self.density is None:
            all_densities = None

        return all_streamlines, all_densities


    def _streamline_from_seeds(self, seeds):

        # We store all streamlines in a single Nx3
        # array, along with an offset specifying the
        # end of each streamline. We over-allocate to
        # the maximum possible number of coordinates
        # we may need to store, and then truncate
        # afterwards.
        maxverts     = 2 * len(seeds) * self.n_per_vox * (self.nsteps + 1)
        streamlines  = np.zeros((maxverts, 3))
        offsets      = np.zeros( len(seeds) * self.n_per_vox, dtype=int)
        pos          = 0
        count        = 0

        for seed, n in it.product(seeds, range(self.n_per_vox)):

            # Track in both directions from
            # the  seed, and merge the result
            seed = self._do_jitter(seed)
            st1  = self._track(seed, forward=True)
            st2  = self._track(seed, forward=False)
            st   = self._merge_streamlines(st1, st2)

            if st is None:
                continue

            npts = len(st)
            if npts >= self.min_n_steps:

                end                  = pos + npts
                streamlines[pos:end] = st
                offsets[    count]   = end
                pos                  = end
                count                = count + 1

        # Truncate to the number of
        # coordinates that were generated
        streamlines = streamlines[:pos]
        offsets     = offsets[:count]

        if len(streamlines) == 0:
            return None, None, None

        # Retrieve density values for
        # every streamline vertex
        if self.filtered_density is not None:
            dens = ndimage.map_coordinates(self.filtered_density,
                                           streamlines.T,
                                           order=self.order)
            dens = dens.reshape(-1, 1)
        else:
            dens = None

        return streamlines, offsets, dens


    def _track(self, seed, forward=True):
        # initial position/orientation
        ix, iy, iz = self._pos_to_voxel(seed)
        v = np.array(self.orientation[ix, iy, iz, :])
        nv = np.linalg.norm(v)
        if nv == 0:
            return None
        v /= nv
        if not forward:
            v *= -1

        st    = np.zeros((self.nsteps + 1, 3))
        st[0] = seed

        step_idx = 1
        while step_idx < self.nsteps + 1:
            cur_pos = st[step_idx - 1]
            ix, iy, iz = self._pos_to_voxel(cur_pos)
            cur_v = np.array(self.orientation[ix, iy, iz, :])
            nv = np.linalg.norm(cur_v)
            if nv == 0:
                break
            cur_v /= nv
            cur_v *= np.dot(cur_v, v)
            next_pos = cur_pos + self.stepvox * cur_v
            if not self._check_position(next_pos):
                break
            st[step_idx] = next_pos
            v = cur_v  # keep track of orientation
            step_idx = step_idx + 1

        st = st[:step_idx]

        if len(st) == 0:
            return None
        else:
            return st[:step_idx]

    def save_trk(self, basename, st, orient, dens=None):
        filename = f"{basename}.trk"
        sys.stdout.write(" - Saving in TRK format...")

        from nibabel.streamlines import TrkFile, Tractogram

        if dens is None:
            data_per_point = None
        else:
            data_per_point = {"densities": dens}

        T = Tractogram(streamlines=st,
                       data_per_point=data_per_point,
                       affine_to_rasmm=self.affine)

        F = TrkFile(T)
        F.header['voxel_to_rasmm'] = self.affine
        F.header['dimensions']     = orient.shape[:3]
        F.save(filename)

        sys.stdout.write(f"DONE: {filename}\n")


    def save_trx(self, basename, st, orient, dens=None):
        filename = f'{basename}.trx'
        sys.stdout.write(" - Saving in TRX format...")

        from trx.trx_file_memmap import TrxFile, save
        from nibabel.streamlines import Tractogram

        if dens is None:
            data_per_point = None
        else:
            data_per_point = {'densities': dens}

        T = Tractogram(streamlines=st,
                       data_per_point=data_per_point,
                       affine_to_rasmm=self.affine)
        T = T.to_world()

        save(TrxFile.from_tractogram(T, orient.header), filename)
        sys.stdout.write(f"DONE: {filename}\n")


    def save_vtk(self, basename, streamlines, dens=None, overlays=[]):
        sys.stdout.write(" - Saving in VTK format...")
        try:
            import vtk
            from vtk.util import numpy_support
        except ImportError:
            print("Failed to import VTK - cannot save in VTK format. Install vtk package to enable")
            return

        polydata = vtk.vtkPolyData()
        lines = vtk.vtkCellArray()
        points = vtk.vtkPoints()

        ptCtr = 0
        for i in range(0, len(streamlines)):
            line = vtk.vtkLine()
            line.GetPointIds().SetNumberOfIds(len(streamlines[i]))
            for j in range(0, len(streamlines[i])):
                point_world = np.dot(self.affine, list(streamlines[i][j]) + [1,])
                points.InsertNextPoint(point_world[:3])
                linePts = line.GetPointIds()
                linePts.SetId(j, ptCtr)

                ptCtr += 1

            lines.InsertNextCell(line)

        polydata.SetLines(lines)
        polydata.SetPoints(points)
        if dens is not None:
            dens = [np.squeeze(d) for d in dens]
            density = numpy_support.numpy_to_vtk(np.hstack(dens))
            polydata.GetPointData().SetScalars(density)

        for fname in overlays:
            print(f" - Saving overlay {fname} in VTI format for Paraview")
            out_fname = os.path.join(os.path.dirname(basename), os.path.basename(fname))
            out_fname = removeExt(out_fname) + '.vti'
            write_image_as_vtk(Image(fname), out_fname)

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(basename + ".vtk")
        writer.SetInputData(polydata)
        writer.Write()

        sys.stdout.write(f"DONE: {basename}.vtk\n")
