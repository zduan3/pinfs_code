import abc
from typing import NamedTuple, Callable, Optional

import torch
from torch.utils.data import Dataset
import numpy as np


class Rays(NamedTuple):
    origins: torch.Tensor
    viewdirs: torch.Tensor

    def foreach(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        return Rays(fn(self.origins), fn(self.viewdirs))

    def to(self, device):
        return Rays(self.origins.to(device), self.viewdirs.to(device))


def get_rays(K: np.ndarray, c2w: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> Rays:
    dirs = torch.stack([(xs - K[0, 2]) / K[0, 0], -(ys - K[1, 2]) / K[1, 1], -torch.ones_like(xs)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize directions
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return Rays(rays_o, rays_d)


def trans_t(t: float):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]], dtype=np.float32)


def rot_phi(phi: float):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]], dtype=np.float32)


def rot_theta(th: float):
    return np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]], dtype=np.float32)


def pose_spherical(theta: float, phi: float, radius: float, rotZ=True, center: np.ndarray = None):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # center: additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    if rotZ:  # swap yz, and keep right-hand
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32) @ c2w

    if center is not None:
        c2w[:3, 3] += center
    return c2w


def intrinsics_from_hwf(H: int, W: int, focal: float):
    return np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ], dtype=np.float32)


class NeRFDataset(Dataset):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, item: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, dict]:
        """returns image(optional for test), pose, intrinsics(optional for test), and extras"""
        pass

    # poses for prediction, usually given by pose_spherical
    def predict_poses(self):
        render_poses = getattr(self, 'render_poses', None)
        return render_poses if isinstance(render_poses, np.ndarray) else None

    # generate predictor from render_poses
    def predictor(self) -> Optional["NeRFPredictor"]:
        poses = self.predict_poses()
        return NeRFPredictor(poses) if poses is not None else None


class NeRFPredictor(NeRFDataset):
    """Given poses for prediction (usually generated from pose_spherical), used for test"""
    def __init__(self, poses: np.ndarray, extra_fn: Callable[[int], dict] = lambda _: {}):
        super().__init__()
        self.poses = poses
        self.extra_fn = extra_fn

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, item):
        # image, intrinsics unavailable
        return None, self.poses[item], None, self.extra_fn(item)

    def predict_poses(self):
        return self.poses

    def predictor(self):
        return self
