"""
fsl-streamlines: Utility functions for VTK output
"""
import numpy as np

def reorient(data, affine):
    """
    Convert Nifti data to orientation required for paraview

    Paraview appears to respect origin and spacing information in VTK
    image data but cannot cope with negative axis directions. So
    we may need to do some axis flips to make sure all these are positive.

    :param data: Numpy array from Nifti file
    :param affine: Nifti voxel->world affine matrix

    :return: Tuple of axis reordering, axis flips, reordered data array, new affine
    """
    transform = affine[:3, :3]
    dim_reorder, dim_flip = [], []
    absmat = np.absolute(transform)
    for dim in range(3):
        newd = np.argmax(absmat[:, dim])
        dim_reorder.append(newd)
        if transform[newd, dim] < 0:
            dim_flip.append(newd)

    if sorted(dim_reorder) != [0, 1, 2]:
        raise RuntimeError("Could not find consistent dimension re-ordering")

    new_data = np.copy(data)
    new_affine = np.copy(affine)

    # Re-order axes
    dim_transpose = list(dim_reorder)
    if len(dim_transpose) < new_data.ndim:
        dim_transpose = dim_transpose + list(range(len(dim_transpose), new_data.ndim))
    new_data = np.transpose(new_data, dim_transpose)
    for idx, dim in enumerate(dim_reorder):
        new_affine[:, dim] = affine[:, idx]

    # Flip dimensions
    for dim in dim_flip:
        new_data = np.flip(new_data, dim)
        new_affine[:, dim] = -new_affine[:, dim]

    # Adjust origin to correct axes flips
    for dim in dim_flip:
        new_affine[:3, 3] = new_affine[:3, 3] - new_affine[:3, dim] * (new_data.shape[dim] - 1)

    return dim_reorder, dim_flip, new_data, new_affine

def write_image_as_vtk(img, fname):
    import vtk
    from vtk.util import numpy_support
    _dim_reorder, _dim_flip, new_data, new_affine = reorient(img.data, img.getAffine('voxel', 'world'))
    data_type = vtk.VTK_FLOAT
    shape = new_data.shape

    # VTK uses fortran order for image data
    flat_data_array = new_data.flatten(order='F')
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])
    img.SetOrigin(new_affine[:3, 3])
    dirs = new_affine[:3, :3].flatten()
    img.SetDirectionMatrix(dirs)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(img)
    writer.Update()
    writer.Write()
