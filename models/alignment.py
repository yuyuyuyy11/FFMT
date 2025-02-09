from typing import Optional

import torch

# from ..utils.transformations import transform_points_Rt
from .model_util import nn_gather

import torch
from pytorch3d import transforms as pt3d_T


@torch.jit.script
def invert_quaternion(quat):
    return quat * torch.tensor([1, -1, -1, -1]).to(quat)


@torch.jit.script
def normalize_quaternion(quat):
    # deal with all 0s
    norm = quat.norm(p=2, dim=-1, keepdim=True)
    w = quat[..., 0:1]
    w = torch.where(norm < 1e-9, w + 1, w)
    quat = torch.cat((w, quat[..., 1:]), dim=-1)

    # normalize
    norm = quat.norm(p=2, dim=-1, keepdim=True)
    quat = quat / quat.norm(p=2, dim=-1, keepdim=True)
    return quat


@torch.jit.script
def quaternion_distance(q0, q1):
    w_rel = (q0 * q1).sum(dim=1)
    w_rel = w_rel.clamp(min=-1, max=1)
    q_rel_error = 2 * w_rel.abs().acos()
    return q_rel_error


@torch.jit.script
def normalize_qt(params):
    t = params[:, 4:7]
    q = params[:, 0:4]
    q = normalize_quaternion(q)
    return torch.cat((q, t), dim=-1)


@torch.jit.script
def apply_quaternion(quaternion, point):
    # Copied over as is from pytorch3d; replaced wiht my invert
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = pt3d_T.quaternion_raw_multiply(
        pt3d_T.quaternion_raw_multiply(quaternion, point_as_quaternion),
        invert_quaternion(quaternion),
    )
    return out[..., 1:]


def random_qt(batch_size: int, q_mag: float, t_mag: float):
    assert q_mag >= 0.0 and q_mag < 1.57, "Rotation angle has to be between 0 and pi/2"

    # Random quaternion of magnitude theta (cos(theta/2
    h_mag = torch.ones(batch_size, 1) * q_mag / 2.0
    q_w = h_mag.cos()
    q_xyz = torch.randn(batch_size, 3)
    q_xyz = (q_xyz / q_xyz.norm(p=2, dim=1, keepdim=True)) * h_mag.sin()

    # get translation
    t = torch.randn(batch_size, 3)
    t = t / t.norm(p=2, dim=1, keepdim=True)

    param = torch.cat((q_w, q_xyz, t), dim=1)
    return param


@torch.jit.script
def transform_points_qt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    N, D = viewpoint.shape
    assert D == 7, "3 translation and 4 quat "

    q = viewpoint[:, None, 0:4]
    t = viewpoint[:, None, 4:7]

    # Normalize quaternion
    q = normalize_quaternion(q)

    if inverse:
        # translate then rotate
        points = points - t
        points = apply_quaternion(invert_quaternion(q), points)
    else:
        # rotate then invert
        points = apply_quaternion(q, points)
        points = points + t

    return points


@torch.jit.script
def transform_points_Rt(
    points: torch.Tensor, viewpoint: torch.Tensor, inverse: bool = False
):
    N, H, W = viewpoint.shape
    assert H == 3 and W == 4, "Rt is B x 3 x 4 "
    t = viewpoint[:, :, 3]
    r = viewpoint[:, :, 0:3]

    # transpose r to handle the fact that P in num_points x 3
    # yT = (RX)T = XT @ RT
    r = r.transpose(1, 2).contiguous()

    # invert if needed
    if inverse:
        points = points - t[:, None, :]
        points = points.bmm(r.inverse())
    else:
        points = points.bmm(r)
        points = points + t[:, None, :]

    return points




def align(corres, P, Q, align_cfg, return_chamfer=False):
    """
    Input:
        corres:  Information for K matches (list)
            idx_1   LongTensor(B, K)        match ids in pointcloud P
            idx_2   LongTensor(B, K)        match ids in pointcloud Q
            dists   FloatTensor(B, K)       match feature cosine distance
        P:          FloatTensor (B, N, 3)   first pointcloud's XYZ
        Q:          FloatTensor (B, N, 3)   second pointcloud's XYZ
        align_cfg:  Alignment config        check config.py MODEL.alignment

    Return:
        FloatTensor (B, 3, 4)       Rt matrix
        FloatTensor (B, )           Weighted Correspondance Error
    """
    # get useful variables
    corr_P_idx, corr_Q_idx, weights, _ = corres

    # get match features and coord
    corr_P = nn_gather(P, corr_P_idx)
    corr_Q = nn_gather(Q, corr_Q_idx)

    Rt = randomized_weighted_procrustes(corr_P, corr_Q, weights, align_cfg)

    # Calculate correspondance loss
    corr_P_rot = transform_points_Rt(corr_P, Rt)
    dist_PQ = (corr_P_rot - corr_Q).norm(p=2, dim=2)

    weights_norm = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    corr_loss = (weights_norm * dist_PQ).sum(dim=1)

    return Rt, corr_loss


def randomized_weighted_procrustes(pts_ref, pts_tar, weights, align_cfg):
    """
    Adapts the Weighted Procrustes algorithm (Choy et al, CVPR 2020) to subsets.
    Specifically, the algorithm randomly samples N subsets and applies the weighted
    procrustes algorithm to it. It then picks the solution that minimzies the chamfer
    distances over all the correspondences.

    Input:
        pts_ref     FloatTensor (N x C x 3)     reference points
        pts_tar     FloatTensor (N x C x 3)     target points
        weights     FloatTensor (N x C)         weights for each correspondance
        align_cfg   YACS config                 alignment configuration

    Returns:        FloatTensor (N x 3 x 4)     Esimated Transform ref -> tar
    """
    # Define/initialize some key variables and cfgs
    batch_size, num_pts, _ = pts_ref.shape

    # Do the SVD optimization on N subsets
    N = align_cfg.num_seeds
    subset = align_cfg.point_ratio

    if subset < 1.0:
        num_matches = int(subset * num_pts)
        indices = torch.LongTensor(batch_size, N, num_matches).to(device=pts_ref.device)
    else:
        num_matches = num_pts

    # get a subset of points and detach
    pts_ref_c = pts_ref.unsqueeze(1).repeat(1, N, 1, 1)
    pts_tar_c = pts_tar.unsqueeze(1).repeat(1, N, 1, 1)
    if subset < 1.0:
        indices.random_(num_pts)
        pts_ref_c = pts_ref_c.gather(2, indices.unsqueeze(3).repeat(1, 1, 1, 3))
        pts_tar_c = pts_tar_c.gather(2, indices.unsqueeze(3).repeat(1, 1, 1, 3))
        if weights is not None:
            weights_c = weights.unsqueeze(1).repeat(1, N, 1)
            weights_c = weights_c.gather(2, indices)
        else:
            weights_c = None

    else:
        if weights is not None:
            weights_c = weights.unsqueeze(1).repeat(1, N, 1)
        else:
            weights_c = None

    # reshape to batch x N --- basically a more manual (and inefficient) vmap, right?!
    pts_ref_c = pts_ref_c.view(batch_size * N, num_matches, 3).contiguous()
    pts_tar_c = pts_tar_c.view(batch_size * N, num_matches, 3).contiguous()
    weights_c = weights_c.view(batch_size * N, num_matches).contiguous()

    # Initialize VP
    Rt = paired_svd(pts_ref_c, pts_tar_c, weights_c)
    Rt = Rt.view(batch_size, N, 3, 4).contiguous()

    best_loss = 1e10 * torch.ones(batch_size).to(pts_ref)
    best_seed = -1 * torch.ones(batch_size).int()

    # Iterate over random subsets/seeds -- should be sped up somehow.
    # We're finding how each estimate performs for all the correspondances and picking
    # the one that achieves the best weighted chamfer error
    for k in range(N):
        # calculate chamfer loss for back prop
        c_Rt = Rt[:, k]
        pts_ref_rot = transform_points_Rt(pts_ref, c_Rt, inverse=False)
        c_chamfer = (pts_ref_rot - pts_tar).norm(dim=2, p=2)

        if weights is not None:
            c_chamfer = weights * c_chamfer

        c_chamfer = c_chamfer.mean(dim=1)

        # Find the better indices, and update best_loss and best_seed
        better_indices = (c_chamfer < best_loss).detach()
        best_loss[better_indices] = c_chamfer[better_indices]
        best_seed[better_indices] = k

    # convert qt to Rt
    Rt = Rt[torch.arange(batch_size), best_seed.long()]
    return Rt


# @torch.jit.script
def paired_svd(X, Y, weights: Optional[torch.Tensor] = None):
    """
    The core part of the (Weighted) Procrustes algorithm. Esimate the transformation
    using an SVD.

    Input:
        X           FloatTensor (B x N x 3)     XYZ for source point cloud
        Y           FloatTensor (B x N x 3)     XYZ for target point cloud
        weights     FloatTensor (B x N)         weights for each correspondeance

    return          FloatTensor (B x 3 x 4)     Rt transformation
    """

    # It's been advised to turn into double to avoid numerical instability with SVD
    X = X.double()
    Y = Y.double()

    if weights is not None:
        eps = 1e-5
        weights = weights.double()
        weights = weights.unsqueeze(2)
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)

        X_mean = (X * weights).sum(dim=1, keepdim=True)
        Y_mean = (Y * weights).sum(dim=1, keepdim=True)
        X_c = weights * (X - X_mean)
        Y_c = weights * (Y - Y_mean)
    else:
        X_mean = X.mean(dim=1, keepdim=True)
        Y_mean = Y.mean(dim=1, keepdim=True)
        X_c = X - X_mean
        Y_c = Y - Y_mean

    # Reflection to handle numerically instable COV matrices
    reflect = torch.eye(3).to(X)
    reflect[2, 2] = -1

    # Calculate H Matrix.
    H = torch.matmul(X_c.transpose(1, 2).contiguous(), Y_c)

    # Compute SVD
    U, S, V = torch.svd(H)

    # Compute R
    U_t = U.transpose(2, 1).contiguous()
    R = torch.matmul(V, U_t)

    # Reflect R for determinant less than 0
    R_det = batch_determinant(R)
    V_ref = torch.matmul(V, reflect[None, :, :])
    R_ref = torch.matmul(V_ref, U_t)
    R = torch.where(R_det[:, None, None] < 0, R_ref, R)

    # Calculate t
    t = Y_mean[:, 0, :, None] - torch.matmul(R, X_mean[:, 0, :, None])
    Rt = torch.cat((R, t[:, :, 0:1]), dim=2)
    return Rt.float()

def batch_determinant(tensor):
    if tensor.shape[1:] != (3, 3):
        raise ValueError("输入张量的形状必须为 (N, 3, 3)")
    
    # 分解每个 3x3 矩阵的元素
    a = tensor[:, 0, 0]
    b = tensor[:, 0, 1]
    c = tensor[:, 0, 2]
    d = tensor[:, 1, 0]
    e = tensor[:, 1, 1]
    f = tensor[:, 1, 2]
    g = tensor[:, 2, 0]
    h = tensor[:, 2, 1]
    i = tensor[:, 2, 2]

    # 使用行列式公式计算每个 3x3 矩阵的行列式
    dets = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    return dets

if __name__ == "__main__":
    rand_pt = torch.randn(4, 1000, 3)

    rand_qt = random_qt(4, 0.5, 3)

    # qt -> rt
    q = rand_qt[:, :4]
    t = rand_qt[:, 4:, None]

    R = pt3d_T.quaternion_to_matrix(q)
    Rinv = pt3d_T.quaternion_to_matrix(pt3d_T.quaternion_invert(q))

    Rti = torch.cat((Rinv, t), dim=2)
    Rt = torch.cat((R, t), dim=2)

    rot_qt = transform_points_qt(rand_pt, rand_qt)
    rot_Rt = transform_points_Rt(rand_pt, Rt)
    rot_Rti = transform_points_Rt(rand_pt, Rti)

    qt_Rt = (rot_qt - rot_Rt).norm(dim=2, p=2).mean()
    qt_Rti = (rot_qt - rot_Rti).norm(dim=2, p=2).mean()
    Rt_Rti = (rot_Rti - rot_Rt).norm(dim=2, p=2).mean()

    print(f"|| points ||:    {rand_pt.norm(p=2,dim=2).mean()}")
    print(f"Diff Rt and qt:  {qt_Rt:.4e}")
    print(f"Diff Rti and qt: {qt_Rti:.4e}")
    print(f"Diff Rti and Rt: {Rt_Rti:.4e}")

# if __name__ == "__main__":
#     t = torch.rand([10,3,3])
#     print(t)
    
#     print(batch_determinant(t))