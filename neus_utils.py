import torch
import torch.nn.functional as F
import numpy as np

from radiance_fields.neus import SDFRadianceField
from nerf.utils import sample_pdf

import os

# fix https://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    import mcubes
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def up_sample(sdf_network: SDFRadianceField, rays_o, rays_d, z_vals, sdf, n_importance, inv_s,
              last=False, inside_sphere_test=True):
    batch_size, n_samples = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
    sdf = sdf.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0)

    if inside_sphere_test:
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        cos_val = cos_val * inside_sphere

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    new_z_vals = sample_pdf(z_vals, weights, n_importance, det=True).detach()

    # cat_z_vals
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    if not last:
        new_sdf = sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf


def sdf2alpha(sdf: torch.Tensor, gradients: torch.Tensor, inv_s: torch.Tensor,
              rays_d: torch.Tensor, dists: torch.Tensor, cos_anneal_ratio: float = 1.0):
    true_cos = (rays_d[..., None, :] * gradients).sum(-1, keepdim=True)
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
    sdf_diff = iter_cos * dists.reshape(*iter_cos.shape) * 0.5

    # Estimate signed distances at section points
    # estimated_next_sdf = sdf + iter_cos * dists.reshape(*iter_cos.shape) * 0.5
    # estimated_prev_sdf = sdf - iter_cos * dists.reshape(*iter_cos.shape) * 0.5

    prev_cdf = torch.sigmoid((sdf - sdf_diff) * inv_s)
    next_cdf = torch.sigmoid((sdf + sdf_diff) * inv_s)

    # p = prev_cdf - next_cdf
    # c = prev_cdf
    # alpha = ((p + 1e-5) / (c + 1e-5)).reshape(*sdf.shape[:-1]).clip(0.0, 1.0)
    alpha = (1.0 - next_cdf / (prev_cdf + 1e-5)).clip(0.0, 1.0)
    return alpha
