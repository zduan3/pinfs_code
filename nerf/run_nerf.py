import os
import time

import numpy as np
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from radiance_fields.nerf import NeRF
from radiance_fields.neus import SDFRadianceField
from datasets.utils import get_rays, intrinsics_from_hwf, NeRFDataset
from nerf.utils import *
from run_pinf_helpers import voxel_output, VoxelTool
from neus_utils import sdf2alpha


# RENDERING

def raw2outputs(raw, z_vals: torch.Tensor, rays_d: torch.Tensor, cos_anneal_ratio: float = 1.0):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        cos_anneal_ratio: for neus
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [n_rays, n_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = raw[0]  # [n_rays, n_samples, 3]
    extra_output = {}
    if 'sdf' in raw[2]:
        extra = raw[2]
        sdf = extra['sdf']
        # sdf = torch.where(mask, sdf, 1e10)  # must larger than max dists
        gradients = extra['gradients']
        norm = torch.linalg.norm(gradients, ord=2, dim=-1)
        # gradients = gradients / norm[..., None]
        inv_s = extra['inv_s']
        alpha = sdf2alpha(sdf, gradients, inv_s, rays_d, dists, cos_anneal_ratio).squeeze(-1)

        extra_output["eikonal_loss"] = ((norm - 1.0) ** 2).mean()
        extra_output['inv_s'] = inv_s
    else:
        alpha = 1.0 - torch.exp(-F.relu(raw[1].squeeze(-1)) * dists)  # [n_rays, n_samples]

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [n_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return NeRFOutputs(rgb_map, depth_map, acc_map, **extra_output), weights


class NeRFRenderer:
    def __init__(self,
                 model: nn.Module,
                 n_samples: int,
                 prop_model: nn.Module = None,
                 n_importance: int = 0,
                 near: float = 0.0,
                 far: float = 1.0,
                 use_viewdirs: bool = False,
                 bg_color: torch.Tensor = None,
                 perturb: bool = False,
                 ):
        """Volumetric rendering.
            Args:
              model: .
              n_samples: int. Number of different times to sample along each ray.
              perturb: bool. If true, each ray is sampled at stratified
                random points in time.
              n_importance: int. Number of additional times to sample along each ray.
                These samples are only passed to network_fine.
              bg_color: 3D Tensor. Background color.
        """
        self.model = model
        self.prop_model = prop_model
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.near = near
        self.far = far
        self.use_viewdirs = use_viewdirs
        self.bg_color = bg_color
        self.perturb = perturb
        self.cos_anneal_ratio = 1.0

    def run(self, rays_o, rays_d, ret_raw=False, perturb: bool = None) -> tuple[NeRFOutputs, ...]:
        if perturb is None:
            perturb = self.perturb

        n_rays = rays_o.shape[0]

        near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])

        t_vals = torch.linspace(0., 1., steps=self.n_samples)
        z_vals = near * (1. - t_vals) + far * t_vals
        z_vals = z_vals.expand([n_rays, self.n_samples])

        if perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]
        if self.use_viewdirs:
            viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        else:
            viewdirs = rays_d   # placeholder

        prop_model = self.prop_model if self.prop_model is not None else self.model
        raw = prop_model.forward(pts, viewdirs[..., None, :].expand(pts.shape))
        out, weights = raw2outputs(raw, z_vals, rays_d)
        out0 = None

        if self.n_importance > 0:
            out0 = out

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1].detach(), self.n_importance, det=not perturb)

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

            # [n_rays, n_samples + n_importance, 3]
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            raw = self.model.forward(pts, viewdirs[..., None, :].expand(pts.shape))

            out, weights = raw2outputs(raw, z_vals, rays_d)

        if ret_raw:
            out.extras['rgb'] = raw[0]
            out.extras['sigma'] = raw[1]

        if self.n_importance > 0:
            return out, out0
        return out,

    def render(self, rays_o, rays_d, chunk, background: torch.Tensor = None, **kwargs):  # -> NeRFOutputs:
        """Render rays
        Args:
            rays_o:
            rays_d: array of shape [2, batch_size, 3]. Ray origin and direction for each example in batch.
            chunk: int. Maximum number of rays to process simultaneously. Used to control maximum memory usage.
                Does not affect final results.
            bkgd_color: Tensor. override bg_color
            kwargs: extra args passed to run()
        Returns:
            rgb_map: [batch_size, 3]. Predicted RGB values for rays.
            disp_map: [batch_size]. Disparity map. Inverse of depth.
            acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
            extras: dict with everything returned by render_rays().
        """
        shape = rays_d.shape[:-1]  # batch_size/shape for input rays and output results

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        # Render and reshape (batchify_rays)
        ret_list = []
        for i in range(0, rays_o.shape[0], chunk):
            ret = self.run(rays_o[i:i + chunk], rays_d[i:i + chunk], **kwargs)
            ret_list.append(ret)

        out = NeRFOutputs.merge([ret[0] for ret in ret_list], shape)
        if background is None:
            background = self.bg_color
        out.add_background(background)
        if len(ret_list[0]) > 1:
            out0 = NeRFOutputs.merge([ret[1] for ret in ret_list], shape, skip_extras=True)
            out0.add_background(background)
            out.extras['coarse'] = out0

        # return out
        return out.rgb, out.depth, out.acc, out.extras


# RUN_NERF

def render_path(renderer: NeRFRenderer, dataset: NeRFDataset, hwf: tuple[int, int, float],
                chunk, save_dir=None, video_prefix=None, render_factor=0, background=None):
    assert save_dir is not None or video_prefix is not None

    H, W, focal = hwf
    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor
    K = intrinsics_from_hwf(H, W, focal)

    rgbs: list[np.ndarray] = []
    render_kwargs = {
        'perturb': False,
        'background': background,
    }

    t = time.time()
    for i, data in enumerate(tqdm(dataset)):
        print(i, time.time() - t)
        t = time.time()
        gt, c2w, _, _ = data
        xs, ys = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
        rays_o, rays_d = get_rays(K, torch.Tensor(c2w), xs, ys)
        rgb, depth, acc, _ = renderer.render(rays_o, rays_d, chunk=chunk, **render_kwargs)
        if i == 0:
            print(rgb.shape)

        rgb8 = to8b(rgb)
        rgbs.append(rgb8)
        if save_dir is not None:
            filename = os.path.join(save_dir, '{:03d}.png'.format(i))
            if gt is not None and render_factor == 0:
                rgb8 = np.concatenate((to8b(gt), rgb8), axis=1)
            imageio.imwrite(filename, rgb8)
            
    if video_prefix is not None:
        print(f'Done, saving to {video_prefix}***.mp4')
        imageio.mimwrite(video_prefix + "rgb.mp4", rgbs, fps=30, quality=8)


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego',
                        help='input data directory')

    # training options
    parser.add_argument("--net_model", type=str, default='nerf',
                        help='which model to use, nerf, siren...')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--n_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--fix_seed", type=int, default=0,
                        help='the random seed.')

    # rendering options
    parser.add_argument("--n_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--n_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=400,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def create_model(name: str, args, input_ch: int = 3):
    D = args.netdepth
    W = args.netwidth
    if name == 'nerf':
        return NeRF(
            D=D, W=W, input_ch=input_ch, use_viewdirs=args.use_viewdirs,
            multires=args.multires, multires_views=args.multires_views
        )
    elif name == 'siren':
        from radiance_fields.siren import SIREN_NeRFt
        return SIREN_NeRFt(
            D=D, W=W, input_ch=input_ch, use_viewdirs=args.use_viewdirs,
        )
    elif name == 'neus':
        from radiance_fields.neus import NeuS
        return NeuS(
            D=D, W=W, input_ch=input_ch, multires=6,
            solid_density=20.0,
        )
    elif name == 'nsr':
        from radiance_fields.nsr import NSR
        return NSR(
            input_ch=input_ch,
            bound=2.0,
        )
    raise NotImplementedError(f"Unknown model name {name}")


def train():
    parser = config_parser()
    args = parser.parse_args()

    set_rand_seed(args.fix_seed)

    # Load data
    if args.dataset_type == 'blender':
        from datasets.blender import BlenderDataset

        train_data = BlenderDataset(args.datadir, args.half_res, white_bkgd=args.white_bkgd, split='train')
        test_data = BlenderDataset(args.datadir, args.half_res, args.testskip, white_bkgd=args.white_bkgd, split='test')

        images_test = test_data.images
        poses_test = test_data.poses
        hwf = images_test.shape[1], images_test.shape[2], train_data.focal
        bkgd_color = torch.tensor(args.white_bkgd, device=device).float()

        # images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', args.datadir)

        near = 2.
        far = 6.

    elif args.dataset_type == 'pinf_data':
        from datasets.pinf import PINFStaticDataset, PINFFrameData

        pinf_data = PINFFrameData(args.datadir, args.half_res)
        frame_id = 0
        train_data = PINFStaticDataset(pinf_data, frame_id=frame_id, split='train')
        test_data = PINFStaticDataset(pinf_data, frame_id=frame_id, split='test')

        all_img_poses = [test_data[i] for i in range(len(test_data))]   # list of (img, pose)
        images_test, poses_test, _, _ = zip(*all_img_poses)
        image = images_test[0]
        hwf = image.shape[0], image.shape[1], train_data.focal
        near, far = pinf_data.near, pinf_data.far
        # near, far = 0.1, 2.0
        bkgd_color = torch.tensor(pinf_data.bkg_color).float()

        print('Loaded pinf', args.datadir)

    elif args.dataset_type == 'nv_data':
        from datasets.pinf import PINFStaticDataset
        from datasets.neural_volumes import PINFFrameDataFromNV

        pinf_data = PINFFrameDataFromNV(args.datadir)

        frame_id = 80
        train_data = PINFStaticDataset(pinf_data, frame_id=frame_id, split='train')
        test_data = PINFStaticDataset(pinf_data, frame_id=frame_id, split='test')

        all_img_poses = [test_data[i] for i in range(len(test_data))]  # list of (img, pose)
        images_test, poses_test, _, _ = zip(*all_img_poses)
        image = images_test[0]
        hwf = image.shape[0], image.shape[1], train_data.focal
        near, far = pinf_data.near, pinf_data.far
        # near, far = 0.1, 2.0
        bkgd_color = torch.tensor(pinf_data.bkg_color).float()

        print('Loaded nv', args.datadir)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    assert isinstance(train_data, NeRFDataset) and isinstance(test_data, NeRFDataset)

    if args.render_test:
        render_data = test_data
    else:
        render_data = test_data.predictor()

    # Create log dir and copy the config file
    expdir: str = os.path.join(args.basedir, args.expname)
    os.makedirs(expdir, exist_ok=True)
    f = os.path.join(expdir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(expdir, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    model = create_model(args.net_model, args)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.n_importance > 0:
        model_fine = model
        if args.net_model == 'neus':
            model = create_model('nerf', args)
        else:
            model = create_model(args.net_model, args)
        grad_vars += list(model.parameters())

    renderer = NeRFRenderer(
        model=model_fine,
        prop_model=model,
        n_samples=args.n_samples,
        n_importance=args.n_importance,
        use_viewdirs=args.use_viewdirs,
        bg_color=bkgd_color,
        perturb=args.perturb,
    )

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(expdir, f) for f in sorted(os.listdir(expdir)) if f.endswith('.tar')]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    # return renderer, start, grad_vars, optimizer
    global_step = start

    renderer.near = near
    renderer.far = far

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images_test
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(expdir,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses', len(render_data))

            render_path(renderer, render_data, hwf, args.chunk,
                        save_dir=testsavedir, video_prefix=os.path.join(testsavedir, 'video_'),
                        render_factor=args.render_factor, background=torch.ones(3))
            return

    # Prepare raybatch tensor if batching random rays
    n_rand = args.n_rand

    n_iters = 200000 + 1
    print('Begin')

    voxel_tran = torch.eye(4, device=device)
    voxel_tran[:3, 3] = torch.tensor([-1.5, -1.5, -1.5], device=device)
    voxel_tran[:3, :3] *= 3.0
    voxel_tran_inv = torch.inverse(voxel_tran)
    voxel_writer = VoxelTool(voxel_tran, voxel_tran_inv, np.ones(3), 64)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    for i in trange(start, n_iters):
        # time0 = time.time()

        # Sample random ray batch
        # Random from one image
        img_i = np.random.choice(len(train_data))
        target, pose, K, extras = train_data[img_i]
        target = torch.Tensor(target).to(device)
        H, W = target.shape[:2]

        background = bkgd_color
        if 'video' in extras:
            from datasets.neural_volumes import VideoDataFromNV
            video = extras.get('video', None)
            if isinstance(video, VideoDataFromNV):
                background = video.background
                background = torch.Tensor(background).to(device)

        xs, ys = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
        rays_o, rays_d = get_rays(K, torch.Tensor(pose), xs, ys)  # (H, W, 3), (H, W, 3)

        if i < args.precrop_iters:
            dH = int(H // 2 * args.precrop_frac)
            dW = int(W // 2 * args.precrop_frac)
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                ), -1)
            if i == start:
                print(
                    f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
        else:
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                 -1)  # (H, W, 2)

        coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0], size=[n_rand], replace=False)  # (n_rand,)
        select_coords = coords[select_inds].long()  # (n_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (n_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (n_rand, 3)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (n_rand, 3)
        if isinstance(background, torch.Tensor) and background.dim() >= 2:  # image background
            background = background[select_coords[:, 0], select_coords[:, 1]]

        #####  Core optimization loop  #####
        rgb, depth, acc, extras = renderer.render(rays_o, rays_d, chunk=args.chunk, ret_raw=True, background=background)
        out0: NeRFOutputs | None = extras.get('coarse', None)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        # trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if out0 is not None:
            img_loss0 = img2mse(out0.rgb, target_s)
            loss = loss + img_loss0
            # psnr0 = mse2psnr(img_loss0)

        eikonal_loss = extras.get('eikonal_loss', 0.0)
        if eikonal_loss > 1e-8:
            loss += eikonal_loss * 0.1

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_print == 0:
            message = f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}"
            if isinstance(model, SDFRadianceField):
                message += f"eikonal: {eikonal_loss:6f}  inv_s: {extras.get('inv_s', 0.0):.6f}"
            tqdm.write(message)

        if i in (1, 100, 400) or i % args.i_img == 0:
            img_dir = os.path.join(expdir, 'vox')
            os.makedirs(img_dir, exist_ok=True)
            with torch.no_grad():
                voxel_output(voxel_writer, model, model_fine, None, img_dir, i, scale=160.0)

        if (i == 2000 or i % args.i_weights == 0) and i > start + 1:
            path = os.path.join(expdir, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': renderer.prop_model.state_dict(),
                'network_fine_state_dict': renderer.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i == 2000 or i % args.i_testset == 0:
            testsavedir = os.path.join(expdir, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses', len(test_data))
            with torch.no_grad():
                render_path(renderer, test_data, hwf, args.chunk, save_dir=testsavedir)
            print('Saved test set')

        if i == 10000 or i % args.i_video == 0:
            moviebase = os.path.join(expdir, 'spiral_{:06d}_'.format(i))
            # Turn on testing mode
            with torch.no_grad():
                render_path(renderer, render_data, hwf, args.chunk, video_prefix=moviebase)

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
