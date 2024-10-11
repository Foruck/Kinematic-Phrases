# Mostly ported from Pytorch3D
# Some functions are adapted from https://github.com/qazwsxal/diffusion-extensions
# Some are self-defined

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Check PYTORCH3D_LICENCE before use

import functools
from typing import Optional

import torch
import torch.nn.functional as F


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


# Added
def matrix_of_angles(cos, sin, inv=False, dim=2):
    assert dim in [2, 3]
    sin = -sin if inv else sin
    if dim == 2:
        row1 = torch.stack((cos, -sin), axis=-1)
        row2 = torch.stack((sin, cos), axis=-1)
        return torch.stack((row1, row2), axis=-2)
    elif dim == 3:
        row1 = torch.stack((cos, -sin, 0*cos), axis=-1)
        row2 = torch.stack((sin, cos, 0*cos), axis=-1)
        row3 = torch.stack((0*sin, 0*cos, 1+0*cos), axis=-1)
        return torch.stack((row1, row2, row3),axis=-2)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def _axis_angle_rotation(axis: str, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles, convention: str):
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = map(_axis_angle_rotation, convention, torch.unbind(euler_angles, -1))
    return functools.reduce(torch.matmul, matrices)


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = torch.randn((n, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.
        requires_grad: Whether the resulting tensor should have the gradient
            flag set.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(
        n, dtype=dtype, device=device, requires_grad=requires_grad
    )
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None, device=None, requires_grad=False
):
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type
        requires_grad: Whether the resulting tensor should have the gradient
            flag set

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device, requires_grad)[0]


def standardize_quaternion(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a, b):
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

# Self-defined
def decompose_axis_angle(axis_angle):
    """
    Decompose axis/angle representation.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Decomposed axis angle represention with axis of shape (..., 3) 
        and angle of shape (...)
    """
    angle  = torch.norm(axis_angle, p=2, dim=-1)
    bottom = 1 / angle
    bottom[bottom.isnan()] = 1.0
    bottom[bottom.isinf()] = 1.0
    axis  = axis_angle * bottom[..., None]
    return axis, angle

# Self-defined
def compose_axis_angle(axis, angle):
    """
    Compose axis angle representation

    Args:
        axis: Axis in axis angle form, as a tensor of shape (..., 3)
        angle: Angle in axis angle form, as a tensor of shape (...)

    Returns:
        Composed axis angle represention of shape (..., 3)
    """
    return axis * angle[..., None]

# Self-defined
def standardize_rotation_6d(d6: torch.Tensor) -> torch.Tensor:
    """
    Standardize 6D rotation representation by Zhou et al. [1]
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of standardized 6D rotation representation of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    
    return torch.cat([b1, b2], dim=-1)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

# Self-defined
def rot_from_to(input: torch.Tensor, src: str, tgt: str) -> torch.Tensor:
    if src == 'aa':
        if tgt == 'aa':
            return input
        elif tgt == 'quat':
            return axis_angle_to_quaternion(input)
        elif tgt == 'rot6d':
            return matrix_to_rotation_6d(axis_angle_to_matrix(input))
        elif tgt == 'matrix':
            return axis_angle_to_matrix(input)
        else:
            raise NotImplementedError
    elif src == 'quat':
        if tgt == 'aa':
            return quaternion_to_axis_angle(input)
        elif tgt == 'quat':
            return input
        elif tgt == 'rot6d':
            return matrix_to_rotation_6d(quaternion_to_matrix(input))
        elif tgt == 'matrix':
            return quaternion_to_matrix(input)
        else:
            raise NotImplementedError
    elif src == 'rot6d':
        if tgt == 'aa':
            return matrix_to_axis_angle(rotation_6d_to_matrix(input))
        elif tgt == 'quat':
            return matrix_to_quaternion(rotation_6d_to_matrix(input))
        elif tgt == 'rot6d':
            return input
        elif tgt == 'matrix':
            return rotation_6d_to_matrix(input)
        else:
            raise NotImplementedError
    elif src == 'matrix':
        if tgt == 'aa':
            return matrix_to_axis_angle(input)
        elif tgt == 'quat':
            return matrix_to_quaternion(input)
        elif tgt == 'rot6d':
            return matrix_to_rotation_6d(input)
        elif tgt == 'matrix':
            return input
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

# Ported from https://github.com/qazwsxal/diffusion-extensions
def skew2vec(skew: torch.Tensor) -> torch.Tensor:
    vec = torch.zeros_like(skew[..., 0])
    vec[..., 0] = skew[..., 2, 1]
    vec[..., 1] = -skew[..., 2, 0]
    vec[..., 2] = skew[..., 1, 0]
    return vec
    
# Ported from https://github.com/qazwsxal/diffusion-extensions
def vec2skew(vec: torch.Tensor) -> torch.Tensor:
    skew = torch.repeat_interleave(torch.zeros_like(vec).unsqueeze(-1), 3, dim=-1)
    skew[..., 2, 1] = vec[..., 0]
    skew[..., 2, 0] = -vec[..., 1]
    skew[..., 1, 0] = vec[..., 2]
    return skew - skew.transpose(-1, -2)

def log_rmat(r_mat: torch.Tensor) -> torch.Tensor:
    '''
    See paper
    Exponentials of skew-symmetric matrices and logarithms of orthogonal matrices
    https://doi.org/10.1016/j.cam.2009.11.032
    For most of the derivatons here
    We use atan2 instead of acos here dut to better numerical stability.
    it means we get nicer behaviour around 0 degrees
    More effort to derive sin terms
    but as we're dealing with small angles a lot,
    the tradeoff is worth it.
    '''
    skew_mat = r_mat - r_mat.transpose(-1, -2)
    sk_vec   = skew2vec(skew_mat)
    s_angle  = sk_vec.norm(p=2, dim=-1) / 2
    c_angle  = (torch.einsum('...ii', r_mat) - 1) / 2
    angle    = torch.atan2(s_angle, c_angle)
    scale    = angle / (2 * s_angle)
    # if s_angle = 0, i.e. rotation by 0 or pi (180), we get NaNs
    # by definition, scale values are 0 if rotating by 0.
    # This also breaks down if rotating by pi, fix further down
    scale[angle == 0.0] = 0.0
    log_r_mat = scale[..., None, None] * skew_mat

    # Check for NaNs caused by 180deg rotations.
    nanlocs = log_r_mat[..., 0, 0].isnan()
    nanmats = r_mat[nanlocs]
    # We need to use an alternative way of finding the logarithm for nanmats,
    # Use eigendecomposition to discover axis of rotation.
    # By definition, these are symmetric, so use eigh.
    # NOTE: linalg.eig() isn't in torch 1.8,
    #       and torch.eig() doesn't do batched matrices
    eigval, eigvec = torch.linalg.eigh(nanmats)
    # Final eigenvalue == 1, might be slightly off because floats, but other two are -ve.
    # this *should* just be the last column if the docs for eigh are true.
    nan_axes  = eigvec[..., -1, :]
    nan_angle = angle[nanlocs]
    nan_skew  = vec2skew(nan_angle[..., None] * nan_axes)
    log_r_mat[nanlocs] = nan_skew
    return log_r_mat

# Adapted from https://github.com/qazwsxal/diffusion-extensions
def rot_lerp(rot_a: torch.Tensor, rot_b: torch.Tensor, weight: torch.Tensor, src: str = 'matrix', tgt: str = 'matrix') -> torch.Tensor:
    ''' Weighted interpolation between rot_a and rot_b
    '''
    # Treat rot_b = rot_a @ rot_c
    # rot_a^-1 @ rot_a = I
    # rot_a^-1 @ rot_b = rot_a^-1 @ rot_a @ rot_c = I @ rot_c
    # once we have rot_c, use axis-angle forms to lerp angle
    rot_a = rot_from_to(rot_a, src, 'matrix')
    rot_b = rot_from_to(rot_b, src, 'matrix')
    rot_c = rot_a.transpose(-1, -2) @ rot_b
    axis, angle = decompose_axis_angle(rot_from_to(rot_c, 'matrix', 'aa'))# rmat_to_aa(rot_c)
    # once we have axis-angle forms, determine intermediate angles.
    # print(weight.shape, angle.shape)
    i_angle     = weight * angle
    aa          = compose_axis_angle(axis, i_angle)
    rot_c_i     = rot_from_to(aa, 'aa', 'matrix')
    res         = rot_from_to(rot_a @ rot_c_i, 'matrix', tgt)
    return res

# Adapted from https://github.com/qazwsxal/diffusion-extensions
def rot_scale(input, scalars, src='matrix', tgt='matrix'):
    '''Scale the magnitude of a rotation,
    e.g. a 45 degree rotation scaled by a factor of 2 gives a 90 degree rotation.

    This is the same as taking matrix powers, but pytorch only supports integer exponents

    So instead, we take advantage of the properties of rotation matrices
    to calculate logarithms easily. and multiply instead.
    '''
    rmat        = rot_from_to(input, src, 'matrix')
    logs        = log_rmat(rmat)
    scaled_logs = logs * scalars[..., None, None]
    out         = torch.matrix_exp(scaled_logs)
    out         = rot_from_to(out, 'matrix', tgt)
    return out

def canonicalize_smplx(poses: torch.Tensor, repr: str, trans: Optional[torch.Tensor] = None, tgt: str = None, return_mat: bool = False):
    '''
    Input: [bs, nframes, njoints, 3/4/6/9]
    '''
    bs, nframes, njoints = poses.shape[:3]
    if tgt is None:
        tgt = repr
    
    global_orient = rot_from_to(poses[:, :, 0], repr, 'matrix')

    # first global rotations
    rot2d = rot_from_to(global_orient[:, 0], 'matrix', 'aa')
    rot2d[:, :2] *= 0  # Remove the rotation along the vertical axis
    rot2d = rot_from_to(rot2d, 'aa', 'matrix')

    # Rotate the global rotation to eliminate Z rotations
    global_orient = torch.einsum("ikj,imkl->imjl", rot2d, global_orient)
    global_orient = rot_from_to(global_orient, 'matrix', tgt)

    # Construct canonicalized version of x
    xc = torch.cat((global_orient[:, :, None], rot_from_to(poses[:, :, 1:], repr, tgt)), dim=2)

    if trans is not None:
        # vel = trans[:, 1:]
        # Turn the translation as well
        trans = torch.einsum("ikj,ilk->ilj", rot2d, trans)
        # trans = torch.cat((torch.zeros(bs, 1, 3, device=vel.device), vel), 1)
        if return_mat:
            return xc, trans, rot2d
        return xc, trans
    else:
        if return_mat:
            return xc, rot2d
        return xc

def rotate_smplx(poses: torch.Tensor, rot2d: torch.Tensor, repr: str, trans: Optional[torch.Tensor] = None, tgt: str = None):
    '''
    Input: [bs, nframes, njoints, 3/4/6/9]
    trans: [bs, nframes, njoints, 3] velocity
    '''
    bs, nframes, njoints = poses.shape[:3]
    if tgt is None:
        tgt = repr
    
    global_orient = rot_from_to(poses[:, :, 0], repr, 'matrix')

    # first global rotations
    # rot2d = rot_from_to(global_orient[:, 0], 'matrix', 'aa')
    # rot2d[:, :2] *= 0  # Remove the rotation along the vertical axis
    # rot2d = rot_from_to(rot2d, 'aa', 'matrix')
    rot2d = rot_from_to(rot2d, repr, 'matrix')  # bz, 3, 3
    global_orient = torch.einsum("ikj,imkl->imjl", rot2d, global_orient)
    global_orient = rot_from_to(global_orient, 'matrix', tgt)

    # Construct canonicalized version of x
    xc = torch.cat((global_orient[:, :, None], rot_from_to(poses[:, :, 1:], repr, tgt)), dim=2)

    if trans is not None:
        # vel = trans[:, 1:]
        # Turn the translation as well
        trans = torch.einsum("ikj,ilk->ilj", rot2d, trans)
        # trans = torch.cat((torch.zeros(bs, 1, 3, device=vel.device), vel), 1)
        return xc, trans
    else:
        return xc
        
# Adapted from https://github.com/zju3dv/EasyMocap/blob/64e0e48d2970b352cfc60ffd95922495083ef306/easymocap/dataset/mirror.py
# TODO
_PERMUTATION = {
    'smpl': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22],
    'smplh': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 24, 25, 23, 24],
    'smplx': [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 24, 25, 23, 24, 26, 28, 27],
    'smplhfull': [
        0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, # body
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
    ],
    'smplxfull': [
        0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, # body
        22, 24, 23,  # jaw, left eye, right eye
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, # right hand
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, # left hand
    ]
}
def flipSMPLXPoses(poses, trans, ptype='smplx'):
    """
        poses: L, 24, 3
        trans: L, 3 
    """
    assert ptype in ['smplx', 'smpl'], '{} is not implemented'.format(ptype)
    # flip pose
    poses = poses[:, _PERMUTATION[ptype + 'full']]
    poses[..., 1:] = -poses[..., 1:]
    trans[:, 0]   *= -1
    
    return poses, trans

# ported from https://github.com/c-he/NeMF/blob/79918430970fd138ae730510459c8f34893a3f86/src/utils.py#L155C1-L171C19
def estimate_linear_velocity(data_seq, dt):
    '''
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order
    differences, respectively
    - h : step size
    '''
    # first steps is forward diff (t+1 - t) / dt
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    # middle steps are second order (t+1 - t-1) / 2dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
    # last step is backward diff (t - t-1) / dt
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq

# ported from https://github.com/c-he/NeMF/blob/main/src/utils.py#L174
def estimate_angular_velocity(rot_seq, dt, repr='matrix'):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    rot_seq = rot_from_to(rot_seq, repr, 'matrix')
    dRdt = estimate_linear_velocity(rot_seq, dt)
    R    = rot_seq
    RT   = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector by averaging symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w   = torch.stack([w_x, w_y, w_z], axis=-1)
    return w