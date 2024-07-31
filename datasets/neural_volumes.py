import os

import numpy as np
import imageio
import cv2

from .pinf import PINFFrameDataBase, VideoData, pose_spherical


def load_krt(path):
    """Load KRT file containing intrinsic and extrinsic parameters."""
    cameras = {}

    with open(path, "r") as f:
        while True:
            name = f.readline()
            if name == "":
                break

            intrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrin = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                    "intrin": np.array(intrin),
                    "dist": np.array(dist),
                    "extrin": np.array(extrin)}

    return cameras


class Dataset:
    def __init__(self, basedir: str, frame_list: list[int],
                 selected_cameras: list[str],
                 world_scale: float = 1.0):
        # krtpath = "experiments/dryice1/data/KRT"
        self.basedir = basedir
        krtpath = os.path.join(basedir, "data/KRT")
        krt = load_krt(krtpath)

        # get options
        self.cameras = sorted(list(krt.keys()))
        self.frame_list = frame_list
        self.selected_cameras = selected_cameras

        # transformation that places the center of the object at the origin
        transfpath = os.path.join(basedir, "data/pose.txt")
        self.transf = np.genfromtxt(transfpath, dtype=np.float32, skip_footer=2)
        # self.transf[:3, :3] *= worldscale

        # compute camera positions
        self.campos, self.camrot, self.focal = {}, {}, {}
        # self.extrin = {}
        self.intrin = {}
        for cam in self.selected_cameras:
            self.campos[cam] = (-np.dot(krt[cam]['extrin'][:3, :3].T, krt[cam]['extrin'][:3, 3])).astype(np.float32)
            self.camrot[cam] = (krt[cam]['extrin'][:3, :3]).astype(np.float32)

            # transform immediately
            self.camrot[cam] = np.dot(self.transf[:3, :3].T, self.camrot[cam].T).T
            self.campos[cam] = np.dot(self.transf[:3, :3].T, self.campos[cam] - self.transf[:3, 3]) * world_scale

            self.focal[cam] = (np.diag(krt[cam]['intrin'][:2, :2]) / 4.).astype(np.float32)
            # self.princpt[cam] = (krt[cam]['intrin'][:2, 2] / 4.).astype(np.float32)
            # self.extrin[cam] = krt[cam]['extrin'].astype(np.float32)
            self.intrin[cam] = krt[cam]['intrin'].astype(np.float32) / 4.


class VideoDataFromNV(VideoData):
    def __init__(self, orig_id):
        super(VideoDataFromNV, self).__init__(None)
        self.intrin: np.ndarray | None = None
        self.orig_id = orig_id
        self.background = None

    def intrinsics(self):
        return self.intrin


def c2w_from_rot_pos(rot: np.ndarray, pos: np.ndarray):
    # rot /= np.sqrt(np.mean(np.diag(rot @ rot.T)))    # normalize rotation
    # t = -rot @ pos
    # t = (t + 1.0) / 2.0
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot.T    # reverse ray dir
    c2w[:3, 1:3] *= -1  # flip z
    c2w[:3, 3] = pos  # (pos + 1.0) / 2.0     # map [-1, 1] to [0, 1]
    return c2w


# PINF adapter
class PINFFrameDataFromNV(PINFFrameDataBase):
    def __init__(self, basedir: str, is_test: bool = False, half_res: str = None, **kwargs):
        super().__init__()

        # black background
        # train_cameras = ["400002", "400007", "400008", "400019", "400023", "400025", "400029", "400030"]
        # test_cameras = ["400037", "400041", "400048", "400064"]
        train_cameras = ["400002", "400008", "400013", "400019", "400023", "400025", "400029", "400030", "400041",
                         # "400035", "400055", "400070", "400010", "400012", "400015", "400016", "400017", "400018"
                         "400004", "400006", "400009", "400010", "400012", "400015", "400016", "400017", "400018",
                         # "400026", "400028", "400035", "400039", "400042", "400053", "400055", "400059", "400060",
                         ]
        # test_cameras = ["400007", "400037", "400048", "400064"]
        test_cameras = ["400007"]
        if is_test:
            train_cameras = train_cameras[:1]  # faster when test only
        nv_data = Dataset(
            basedir=basedir,
            frame_list=[i for i in range(15469, 16578, 3)][:-1],
            selected_cameras=train_cameras + test_cameras,
            world_scale=1. / 256
        )

        # map [-1, 1] to [0, 1]
        self.voxel_tran = np.eye(4, dtype=np.float32)
        self.voxel_tran[:3, 3] = -1.0
        self.voxel_scale = 2.0 * np.ones(3, dtype=np.float32)

        half_ratio = None
        if half_res == 'half':
            half_ratio = 2
        elif half_res == 'quarter':
            half_ratio = 4
        elif half_res is not None:
            if half_res != 'normal':
                print("Unsupported half_res value", half_res)

        videos = []
        # frame_list = nv_data.frame_list[:len(nv_data.frame_list) // 2]  # takes first half
        frame_list = nv_data.frame_list[:150]
        for cam in nv_data.selected_cameras:   # gather frames from one camera as video
            video = VideoDataFromNV(cam)
            video.transform_matrix = c2w_from_rot_pos(nv_data.camrot[cam], nv_data.campos[cam])
            frames = []
            for frame in frame_list:
                imagepath = os.path.join(nv_data.basedir, "data/cam{}/image{:04}.jpg".format(cam, int(frame)))
                image = imageio.v2.imread(imagepath)

                if half_ratio is not None:
                    H, W = image.shape[0] // half_ratio, image.shape[1] // half_ratio
                    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

                frames.append(image)
            bg_path = os.path.join(nv_data.basedir, "data/cam{}/bg.jpg".format(cam))
            if os.path.exists(bg_path):
                video.background = imageio.v2.imread(bg_path)
                if half_ratio is not None:
                    H, W = video.background.shape[0] // half_ratio, video.background.shape[1] // half_ratio
                    video.background = cv2.resize(video.background, (W, H), interpolation=cv2.INTER_AREA)
            video.frames = np.array(frames, dtype=np.float32) / 255.0
            video.background = np.array(video.background, dtype=np.float32) / 255.0
            video.focal = nv_data.focal[cam][0]
            video.intrin = nv_data.intrin[cam]
            if half_ratio is not None:
                video.focal /= half_ratio
                video.intrin /= half_ratio
            video.delta_t = 1.0 / len(frames)
            videos.append(video)
        self.videos = {
            'train': videos[:len(train_cameras)],
            'test': videos[len(train_cameras):]
        }
        self.t_info = np.float32([0.0, 1.0, 1.0 / len(videos[0])])

        # set render settings:
        sp_n = 40  # an even number!
        sp_poses = [
            pose_spherical(angle, phi=-30.0, radius=4.0, rotZ=False, center=np.zeros(3, dtype=np.float32))
            for angle in np.linspace(-180, 180, sp_n + 1)[:-1]
        ]
        self.render_poses = np.stack(sp_poses, 0)  # [sp_poses[36]]*sp_n, for testing a single pose
        self.render_timesteps = np.linspace(self.t_info[0], self.t_info[1], num=sp_n).astype(np.float32)
        self.bkg_color = np.zeros(3, dtype=np.float32)
        self.near, self.far = 3.0, 5.0
        # self.near, self.far = 1.0, 3.0

        # self.nv_data = nv_data  # backup
