import torch
from src.utils.libkdtree import KDTree
import numpy as np
import math
import pdb


def ChangeBasis(plane_parameters, device='cuda'):
    '''
    C_mat is stacked matrices of:
    1. Change of basis matrices (batch_size x L x 3 x 3)
    2. Normalizing constants (batch_size x L x 1 x 3)

    Args:
            plane_parameters (tensor) : Plane parameters (batch_size x L x 3) - torch.tensor dtype = torch.float32
    Output:
            C_mat (tensor) : (batch_size x L x 4 x 3)
    '''
    device = device
    batch_size, L, _ = plane_parameters.size()
    normal = plane_parameters.reshape([batch_size * L, 3]).float().to(device)
    normal = normal / torch.norm(normal, p=2, dim=1).view(batch_size * L, 1)  # normalize
    normal = normal + 0.000001  # Avoid non-invertible matrix down the road

    basis_x = torch.tensor([1, 0, 0], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
    basis_y = torch.tensor([0, 1, 0], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
    basis_z = torch.tensor([0, 0, 1], dtype=torch.float32).repeat(batch_size * L, 1).to(device)

    v = torch.cross(basis_z.to(device), normal, dim=1)
    zero = torch.zeros([batch_size * L], dtype=torch.float32).to(device)
    skew = torch.zeros([batch_size * L, 3, 3], dtype=torch.float32).to(device)
    skew[range(batch_size * L), 0] = torch.stack([zero, -v[:, 2], v[:, 1]]).t()
    skew[range(batch_size * L), 1] = torch.stack([v[:, 2], zero, -v[:, 0]]).t()
    skew[range(batch_size * L), 2] = torch.stack([-v[:, 1], v[:, 0], zero]).t()

    idty = torch.eye(3).to(device)
    idty = idty.reshape((1, 3, 3))
    idty = idty.repeat(batch_size * L, 1, 1)
    dot = (1 - torch.sum(normal * basis_z, dim=1)).unsqueeze(1).unsqueeze(2)
    div = torch.norm(v, p=2, dim=1) ** 2
    div = div.unsqueeze(1).unsqueeze(2)

    R = (idty + skew + torch.matmul(skew, skew) * dot / div)

    new_basis_x = torch.bmm(R, basis_x.unsqueeze(2))
    new_basis_y = torch.bmm(R, basis_y.unsqueeze(2))
    new_basis_z = torch.bmm(R, basis_z.unsqueeze(2))

    new_basis_matrix = torch.cat([new_basis_x, new_basis_y, new_basis_z], dim=2)

    C_inv = torch.inverse(new_basis_matrix)

    # Define normalization constant
    b_x = torch.abs(new_basis_x).squeeze(2)
    b_y = torch.abs(new_basis_y).squeeze(2)
    p_dummy = torch.tensor([1, 1, 1], dtype=torch.float32).repeat(batch_size * L, 1).to(device)
    p_x = torch.sum(b_x * p_dummy, dim=1).unsqueeze(1) / torch.sum(b_x * b_x, dim=1).unsqueeze(1) * b_x
    p_y = torch.sum(b_y * p_dummy, dim=1).unsqueeze(1) / torch.sum(b_y * b_y, dim=1).unsqueeze(1) * b_y

    c_x = torch.norm(p_x, p=2, dim=1)
    c_y = torch.norm(p_y, p=2, dim=1)

    normalizer = torch.max(c_x, c_y).unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)
    C_mat = torch.cat([C_inv, normalizer], dim=1)
    C_mat = C_mat.view(batch_size, L, 4, 3)

    return C_mat


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def chamfer_distance(points1, points2, use_kdtree=True, give_id=False):
    ''' Returns the chamfer distance for the sets of points.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        use_kdtree (bool): whether to use a kdtree
        give_id (bool): whether to return the IDs of nearest points
    '''
    if use_kdtree:
        return chamfer_distance_kdtree(points1, points2, give_id=give_id)
    else:
        return chamfer_distance_naive(points1, points2)


def chamfer_distance_naive(points1, points2):
    ''' Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set    
    '''
    assert(points1.size() == points2.size())
    batch_size, T, _ = points1.size()

    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)

    distances = (points1 - points2).pow(2).sum(-1)

    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)

    chamfer = chamfer1 + chamfer2
    return chamfer


def chamfer_distance_kdtree(points1, points2, give_id=False):
    ''' KD-tree based implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    batch_size = points1.size(0)

    # First convert points to numpy
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()

    # Get list of nearest neighbors indieces
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indieces
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)

    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances


def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def transform_points(points, transform):
    ''' Transforms points with regard to passed camera information.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert(points.size(2) == 3)
    assert(transform.size(1) == 3)
    assert(points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points @ R.transpose(1, 2) + t.transpose(1, 2)
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ K.transpose(1, 2)

    return points_out


def b_inv(b_mat):
    ''' Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    '''

    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv


def transform_points_back(points, transform):
    ''' Inverts the transformation.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    assert(points.size(2) == 3)
    assert(transform.size(1) == 3)
    assert(points.size(0) == transform.size(0))

    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points - t.transpose(1, 2)
        points_out = points_out @ b_inv(R.transpose(1, 2))
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ b_inv(K.transpose(1, 2))

    return points_out


def project_to_camera(points, transform):
    ''' Projects points to the camera plane.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    '''
    p_camera = transform_points(points, transform)
    p_camera = p_camera[..., :2] / p_camera[..., 2:]
    return p_camera


def fix_Rt_camera(Rt, loc, scale):
    ''' Fixes Rt camera matrix.

    Args:
        Rt (tensor): Rt camera matrix
        loc (tensor): location
        scale (float): scale
    '''
    # Rt is B x 3 x 4
    # loc is B x 3 and scale is B
    batch_size = Rt.size(0)
    R = Rt[:, :, :3]
    t = Rt[:, :, 3:]

    scale = scale.view(batch_size, 1, 1)
    R_new = R * scale
    t_new = t + R @ loc.unsqueeze(2)

    Rt_new = torch.cat([R_new, t_new], dim=2)

    assert(Rt_new.size() == (batch_size, 3, 4))
    return Rt_new

def normalize_coordinate(p, padding=0.1, plane='xz'):
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    # xy_new = (xy - min_xy) / (max_xy - min_xy + 10e-6) * reso

    xy_new = xy / (1 + padding + 10e-6) # make coordinate back to (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # There are some outliers out of the range of (0, 1)
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_dynamic_plane_coordinate(p, change_basis_matrix_normalizer, padding=0.1):
    device = 'cuda' if (torch.cuda.is_available() == True) else 'cpu'    
    change_basis_matrix = change_basis_matrix_normalizer[:,:3]
    normalizer = (change_basis_matrix_normalizer[:,3,0] + 0.05).to(device)

    max_dim = (1.0 + padding)/2
    p = torch.div(p, max_dim).to(device) # range (-1.0, 1.0)

    p = torch.transpose(p, 2, 1)
    p = torch.bmm(change_basis_matrix.to(device), p)
    p = torch.transpose(p, 2, 1)

    p = p / normalizer.unsqueeze(1).unsqueeze(2) # range (-1.0, 1.0)
    xy = p[:, :, [0, 1]]
    xy_new = xy / 2 # make coordinate back to (-0.5, 0.5)
    xy_new = xy_new + 0.5  # range (0, 1)

    #xy_new = xy / (1 + padding + 10e-6) # make coordinate back to (-0.5, 0.5)
    #xy_new = xy_new + 0.5 # range (0, 1)

    # There are some outliers out of the range of (0, 1)
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def normalize_3d_coordinate(p, padding=0.1):
    
    p_nor = p / (1 + padding + 10e-4) # make coordinate back to (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # There are some outliers out of the range of (0, 1)
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor

def coordinate2index(x, reso, coord_type='2d'):
    x = (x * reso).long()
    if coord_type == '2d': # under the resolution of ground plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # under the resolution of defined grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index



class positional_encoding_prev(object):
    def __init__(self):
        super().__init__()
        L = 10
        freq_bands = 2.**(np.linspace(0, L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * p))
            out.append(torch.cos(freq * p))

        p = torch.cat(out, dim=2)
        return p


class positional_encoding(object):
    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function
        L = 10
        freq_bands = 2.**(np.linspace(0,L-1, L))
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0 # change to the range [-1, 1]

        for freq in self.freq_bands:
            out.append(torch.sin(freq * p))
            out.append(torch.cos(freq * p))

        p = torch.cat(out, dim=2)
        return p
