import numpy as np
import time
from numba import jit
import cupy as cp
from pysingfel.util import deprecated

from . import convert, mapping


@deprecated("Please use 'take_slice' or 'extract_slice' instead. "
            "Note, however, than the signature is different!")
def take_one_slice(local_index, local_weight, volume, pixel_num, pattern_shape):
    """
    Take one slice from the volume given the index and weight and some
    other information.

    :param local_index: The index containing values to take.
    :param local_weight: The weight for each index
    :param volume: The volume to slice from
    :param pixel_num: pixel number.
    :param pattern_shape: The shape of the pattern
    :return: The slice.
    """
    return extract_slice(local_index, local_weight, volume)


def extract_slice(local_index, local_weight, volume):
    """
    Take one slice from the volume given the index and weight map.

    :param local_index: The index containing values to take.
    :param local_weight: The weight for each index.
    :param volume: The volume to slice from.
    :return: The slice.
    """
    # Convert the index of the 3D diffraction volume to 1D
    pattern_shape = local_index.shape[:3]
    pixel_num = np.prod(pattern_shape)

    volume_num_1d = volume.shape[0]
    convertion_factor = np.array(
        [volume_num_1d * volume_num_1d, volume_num_1d, 1], dtype=np.int64)

    index_2d = np.reshape(local_index, [pixel_num, 8, 3])
    index_2d = np.matmul(index_2d, convertion_factor)

    volume_1d = np.reshape(volume, volume_num_1d ** 3)
    weight_2d = np.reshape(local_weight, [pixel_num, 8])

    # Expand the data to merge
    data_to_merge = volume_1d[index_2d]

    # Merge the data
    data_merged = np.sum(np.multiply(weight_2d, data_to_merge), axis=-1)

    return np.reshape(data_merged, pattern_shape)


def take_slice(volume, voxel_length, pixel_momentum, orientation,
               inverse=False):
    """
    Take 1 slice.

    :param volume: The volume to slice from.
    :param voxel_length: The length unit of the voxel
    :param pixel_momentum: The coordinate of each pixel in the reciprocal space measured in A
    :param orientation: The orientation of the slice.
    :param inverse: Whether to use the inverse of the rotation or not.
    :return: The slices.
    """
    # construct the rotation matrix
    rot_mat = convert.quaternion2rot3d(orientation)
    if inverse:
        rot_mat = cp.linalg.inv(rot_mat)

    # rotate the pixels in the reciprocal space.
    # Notice that at this time, the pixel position is in 3D
    rotated_pixel_position = mapping.rotate_pixels_in_reciprocal_space(
        rot_mat, pixel_momentum)
    # calculate the index and weight in 3D
    index, weight = mapping.get_weight_and_index(
        pixel_position=rotated_pixel_position,
        voxel_length=voxel_length,
        voxel_num_1d=volume.shape[0])
    # get one slice
    return extract_slice(local_index=index,
                         local_weight=weight,
                         volume=volume)


@deprecated("The typo was corrected. Please use 'take_n_slices' instead. "
            "Note, however, than the signature is different!")
def take_n_slice(pattern_shape, pixel_momentum,
                 volume, voxel_length, orientations, inverse=False):
    return take_n_slices(volume, voxel_length, pixel_momentum, orientations,
                         inverse)


def take_n_slices(volume, voxel_length, pixel_momentum, orientations,
                  inverse=False):
    """
    Take several slices.

    :param volume: The volume to slice from.
    :param voxel_length: The length unit of the voxel
    :param pixel_momentum: The coordinate of each pixel in the
        reciprocal space measured in A.
    :param orientations: The orientations of the slices.
    :param inverse: Whether to use the inverse of the rotation or not.
    :return: n slices.
    """
    # Preprocess
    slice_num = orientations.shape[0]
    pattern_shape = pixel_momentum.shape[:-1]

    # Create variable to hold the slices
    slices_holder = np.zeros((slice_num,) + pattern_shape, dtype=volume.dtype)

    for l in range(slice_num):
        slices_holder[l] = take_slice(volume, voxel_length, pixel_momentum,
                                      orientations[l], inverse)

    return slices_holder


def cpo_take_n_slices(volume, voxel_length, pixel_momentum, orientations,
                  inverse=False):
    """
    Take several slices.

    :param volume: The volume to slice from.
    :param voxel_length: The length unit of the voxel
    :param pixel_momentum: The coordinate of each pixel in the
        reciprocal space measured in A.
    :param orientations: The orientations of the slices.
    :param inverse: Whether to use the inverse of the rotation or not.
    :return: n slices.
    """
    # Preprocess
    slice_num = orientations.shape[0]
    pattern_shape = pixel_momentum.shape[:-1]

    # Create variable to hold the slices
    slices_holder = np.zeros((slice_num,) + pattern_shape, dtype=volume.dtype)

    for l in range(slice_num):
        slices_holder[l] = cpo_take_slice(volume, voxel_length, pixel_momentum,
                                      orientations[l], inverse)
    #slices_holder[:slice_num] = cpo_take_slice(volume, voxel_length, pixel_momentum,
    #                                  orientations[:slice_num], inverse)
    return slices_holder


# From ps.take_slice, unwrap get_weight_and_index and extract_slice
def cpo_take_slice(volume, voxel_length, pixel_momentum, orientation, inverse=False):
    vn = volume.shape[0]
    vn2 = vn ** 2

    volume = cp.reshape(cp.array(volume), vn**3)
    # construct the rotation matrix
    rot_mat = convert.quaternion2rot3d(orientation)
    rot_mat = cp.array(rot_mat)
    if inverse:
        rot_mat = cp.linalg.inv(rot_mat)

    # rotate the pixels in the reciprocal space.
    # Notice that at this time, the pixel position is in 3D
    pixel_momentum = cp.array(pixel_momentum)

    pixel_position = cp.dot(pixel_momentum, rot_mat.T)
    
    # Extract the detector shape
    detector_shape = pixel_position.shape[:-1]
    pixel_num = np.prod(detector_shape)
    
    # Cast the position infor to the shape [pixel number, 3]
    pixel_position_1d = cp.reshape(pixel_position, (pixel_num, 3))

    # convert_to_voxel_unit
    pixel_position_1d_voxel_unit = pixel_position_1d / voxel_length

    # shift the center position
    shift = (vn - 1) / 2.
    pixel_position_1d_voxel_unit += shift

    # Get one nearest neighbor
    tmp_index = cp.floor(pixel_position_1d_voxel_unit).astype(np.int64)

    # Generate the holders
    indexes = cp.zeros((pixel_num, 8), dtype=np.int64)
    weight = cp.ones((pixel_num, 8), dtype=np.float64)

    # Calculate the floors and the ceilings
    dfloor = pixel_position_1d_voxel_unit - tmp_index
    dceiling = 1 - dfloor

    # Assign the correct values to the indexes
    indexes[:, 0] = vn2 * (tmp_index[:, 0]) + vn * (tmp_index[:, 1]) + (tmp_index[:, 2])
    indexes[:, 1] = vn2 * (tmp_index[:, 0]) + vn * (tmp_index[:, 1]) + (tmp_index[:, 2]+1)
    indexes[:, 2] = vn2 * (tmp_index[:, 0]) + vn * (tmp_index[:, 1]+1) + (tmp_index[:, 2])
    indexes[:, 3] = vn2 * (tmp_index[:, 0]) + vn * (tmp_index[:, 1]+1) + (tmp_index[:, 2]+1)
    indexes[:, 4] = vn2 * (tmp_index[:, 0]+1) + vn * (tmp_index[:, 1]) + (tmp_index[:, 2])
    indexes[:, 5] = vn2 * (tmp_index[:, 0]+1) + vn * (tmp_index[:, 1]) + (tmp_index[:, 2]+1)
    indexes[:, 6] = vn2 * (tmp_index[:, 0]+1) + vn * (tmp_index[:, 1]+1) + (tmp_index[:, 2])
    indexes[:, 7] = vn2 * (tmp_index[:, 0]+1) + vn * (tmp_index[:, 1]+1) + (tmp_index[:, 2]+1)

    # Assign the correct values to the weight
    weight[:, 0] = cp.prod(dceiling, axis=-1)
    weight[:, 1] = dceiling[:, 0] * dceiling[:, 1] * dfloor[:, 2]
    weight[:, 2] = dceiling[:, 0] * dfloor[:, 1] * dceiling[:, 2]
    weight[:, 3] = dceiling[:, 0] * dfloor[:, 1] * dfloor[:, 2]
    weight[:, 4] = dfloor[:, 0] * dceiling[:, 1] * dceiling[:, 2]
    weight[:, 5] = dfloor[:, 0] * dceiling[:, 1] * dfloor[:, 2]
    weight[:, 6] = dfloor[:, 0] * dfloor[:, 1] * dceiling[:, 2]
    weight[:, 7] = cp.prod(dfloor, axis=-1)
    
    # Expand the data to merge
    data_to_merge = volume[indexes]

    # Merge the data
    data_merged = cp.sum(cp.multiply(weight, data_to_merge), axis=-1)

    return cp.asnumpy(cp.reshape(data_merged, detector_shape))
