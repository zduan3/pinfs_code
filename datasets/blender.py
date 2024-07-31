import numpy as np
import os
import json
import imageio
import cv2

from .utils import pose_spherical, NeRFDataset, intrinsics_from_hwf


class BlenderDataset(NeRFDataset):
    def __init__(self, basedir: str, half_res: bool = False, test_skip: int = 1, white_bkgd: bool = False,
                 split: str = 'train'):
        super().__init__()
        with open(os.path.join(basedir, f'transforms_{split}.json'), 'r') as fp:
            meta = json.load(fp)

        imgs = []
        poses = []
        training = split == 'train'
        skip = 1 if training else test_skip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.v2.imread(fname))
            poses.append(frame['transform_matrix'])
        imgs = np.array(imgs, dtype=np.float32) / 255.0  # keep all 4 channels (RGBA)
        if white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1.0 - imgs[..., -1:])
        else:
            imgs = imgs[..., :3]
        poses = np.array(poses, dtype=np.float32)

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)

        if not training:
            render_poses = np.stack(
                [pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]])
        else:
            render_poses = None

        if half_res:
            H = H // 2
            W = W // 2
            focal = focal / 2.

            imgs_half_res = np.empty((imgs.shape[0], H, W, 3), dtype=np.float32)
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res

        self.images = imgs
        self.focal = focal
        self.color_bkgd = np.ones(3) if white_bkgd else np.zeros(3)

        self.poses = poses
        self.render_poses = render_poses

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        return image, self.poses[item], intrinsics_from_hwf(image.shape[0], image.shape[1], self.focal), {}
