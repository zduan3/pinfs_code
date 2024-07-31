import os
import time
import sys
from datetime import datetime
import shutil

import cv2
import numpy as np
import torch
import imageio
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from radiance_fields import RadianceField, HybridRadianceField
from radiance_fields.nerf import NeRF
from radiance_fields.siren import SIREN_NeRFt, SIREN_vel
from radiance_fields.neus import SDFRadianceField, NeuS
from datasets import get_rays
from datasets.utils import intrinsics_from_hwf, NeRFDataset
from datasets.pinf import PINFFrameData, PINFDataset, PINFTestDataset
from nerf.utils import *

from run_pinf_helpers import *
# vel_uv2hsv, den_scalar2rgb, jacobian3D, jacobian3D_np
# vel_world2smoke, vel_smoke2world, pos_world2smoke, pos_smoke2world
# Logger, VGGlossTool, ghost_loss_func
from vgg_tools import VGGLossTool, vgg_sample
from pinf_rendering import PINFRenderer


def render_path(renderer: PINFRenderer, dataset: NeRFDataset, hwf: tuple[int, int, float],
                chunk, save_dir=None, video_prefix=None, render_factor=0, bkgd_color=None, lpips_fn=None):
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
        'ignore_vel': True,
    }

    t = time.time()
    stat_lpips, stat_ssim, stat_psnr = [], [], []
    for i, data in enumerate(tqdm(dataset)):
        print(i, time.time() - t)
        t = time.time()

        gt, c2w, _, ex = data
        cur_timestep = ex["timestep"]

        xs, ys = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
        rays_o, rays_d = get_rays(K, torch.Tensor(c2w), xs, ys)
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        output = renderer.render(rays_o, rays_d, chunk=chunk, timestep=cur_timestep,
                                 background=bkgd_color, **render_kwargs)
        if i == 0:
            print(output.rgb.shape)

        rgb8 = to8b(output.rgb)
        # rgbs.append(rgb8)

        if gt is not None:
            print()
            if lpips_fn is not None:
                stat_lpips.append(lpips_fn(torch.tensor(gt, device=output.rgb.device), output.rgb).item())
                print(f"LPIPS[{i}] = {stat_lpips[-1]}")

            gt = to8b(gt)
            ssim = get_MSSIM(gt, rgb8)
            stat_ssim.append(np.mean(ssim))
            stat_psnr.append(cv2.PSNR(gt, rgb8))
            print(f"SSIM[{i}] = {stat_ssim[-1]}")
            print(f"PSNR[{i}] = {stat_psnr[-1]}")

        if save_dir is not None:
            # filename = os.path.join(save_dir, '{:03d}.png'.format(i))
            # imageio.imwrite(filename, rgb8)

            other_rgbs = []
            out_maps = {}
            if gt is not None and render_factor == 0:
                out_maps['gt'] = gt
                # other_rgbs.append(gt)
            # other_rgbs.append(rgb8)
            out_maps['rgb'] = rgb8
            extras = output.extras
            for rgb_i in ['static', 'dynamic', 'coarse']:
                out = extras.get(rgb_i)
                if isinstance(out, NeRFOutputs):
                    out_maps[rgb_i] = to8b(out.rgb)
                    # other_rgbs.append(to8b(out.rgb))

            grad = extras.get('grad_map')
            if grad is not None:
                grad = grad.cpu().numpy()

                rot = np.linalg.inv(c2w[:3, :3])
                normals = np.matmul(rot, grad[..., None]).squeeze()
                normals = to8b(normals * 0.5 + 0.5)
                out_maps['grad'] = normals
                # if len(other_rgbs) == 4:
                #     other_rgbs = other_rgbs[:-1]    # drop coarse
                # other_rgbs.append(normals)
            # else:
            #     other_rgbs = other_rgbs[:-1]    # drop coarse

            # keys = 'gt', 'rgb', 'static', 'dynamic', 'coarse', 'grad'
            if gt is None:
                keys = 'rgb', 'dynamic', 'static', 'grad'
            else:
                keys = 'gt', 'rgb', 'static', 'dynamic'
            two_cols = False

            for rgb_i in keys:
                if rgb_i in out_maps:
                    other_rgbs.append(out_maps[rgb_i])

            if len(other_rgbs) >= 1:
                if len(other_rgbs) % 2 == 0 and two_cols:
                    hlen = len(other_rgbs) // 2
                    other_rgbs = np.vstack([np.hstack(other_rgbs[:hlen]), np.hstack(other_rgbs[hlen:])])
                else:
                    other_rgbs = np.concatenate(other_rgbs, axis=1)
                filename = os.path.join(save_dir, '_{:03d}.png'.format(i))
                imageio.imwrite(filename, other_rgbs)

                rgbs.append(other_rgbs)

    if len(stat_psnr) > 0:
        if lpips_fn is not None:
            print(f"LPIPS avg. = {np.average(stat_lpips)}")
        print(f"SSIM avg. = {np.average(stat_ssim)}")
        print(f"PSNR avg. = {np.average(stat_psnr)}")

    if video_prefix is not None and len(rgbs) >= 30:
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

    # data info
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # blender flags
    parser.add_argument("--half_res", type=str, default='normal',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # network arch
    parser.add_argument("--net_model", type=str, default='nerf',
                        help='which model to use, nerf, siren...')
    parser.add_argument("--s_model", type=str, default='',
                        help='which model to use for static part, nerf, siren, neus...')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--multires", type=int, default=0,  # 10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=0,  # 4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--omega", type=float, default=30.0,
                        help="first_omega_0 in SIREN")
    parser.add_argument("--use_first_omega", action="store_true",
                        help="enable is_first in SIREN")
    parser.add_argument("--vel_no_slip", action='store_true',
                        help="use no-slip boundray in velocity training")
    parser.add_argument("--use_color_t", action='store_true',
                        help="use time input in static part's color net")

    # network save load
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--fix_seed", type=int, default=42,
                        help='the random seed.')

    # train params - sampling
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk", type=int, default=4096,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--train_warp", default=False, action='store_true',
                        help='train radiance model with velocity warpping')
    parser.add_argument("--vol_output_W", type=int, default=256,
                        help='In output mode: the output resolution along x; In training mode: the sampling resolution for training')

    # train params - iterations
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--fading_layers", type=int, default=-1,
                        help='for siren and hybrid models, the step to finish fading model layers one by one during training.')
    parser.add_argument("--tempo_fading", type=int, default=2000,
                        help='for hybrid model, how many steps try to use static model to represent whole scene')
    parser.add_argument("--vel_delay", type=int, default=10000,
                        help='for siren and hybrid models, the step to start learning the velocity.')
    parser.add_argument("--N_iter", type=int, default=200000,
                        help='for siren and hybrid models, the step to start learning the velocity.')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=400,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=2000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=25000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    # train params - loss weights
    parser.add_argument("--vgg_strides", type=int, default=4,
                        help='vgg stride, should >= 2')
    parser.add_argument("--ghostW", type=float,
                        default=-0.0, help='weight for the ghost density regularization')
    parser.add_argument("--ghost_scale", type=float,
                        default=4.0, help='tolerance for the ghost density regularization')
    parser.add_argument("--vggW", type=float,
                        default=-0.0, help='weight for the VGG loss')
    parser.add_argument("--overlayW", type=float,
                        default=-0.0, help='weight for the overlay regularization')
    parser.add_argument("--nseW", type=float,
                        default=0.001, help='velocity model, training weight for the physical equations')
    parser.add_argument("--eikonal", type=float,
                        default=0.01, help='weight for eikonal loss')
    parser.add_argument("--devW", type=float,
                        default=0.0, help='weight for deviation loss')
    parser.add_argument("--neumann", type=float,
                        default=0.0, help='weight for neumann loss')

    # scene params
    parser.add_argument("--bbox_min", type=str,
                        default='', help='use a boundingbox, the minXYZ')
    parser.add_argument("--bbox_max", type=str,
                        default='1.0,1.0,1.0', help='use a boundingbox, the maxXYZ')
    parser.add_argument("--near", type=float,
                        default=-1.0, help='near plane in rendering, <0 use scene default')
    parser.add_argument("--far", type=float,
                        default=-1.0, help='far plane in rendering, <0 use scene default')

    # task params
    parser.add_argument("--vol_output_only", action='store_true',
                        help='do not optimize, reload weights and output volumetric density and velocity')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    return parser


def prepare_logging(expdir, args):
    # logs
    os.makedirs(expdir, exist_ok=True)
    date_str = datetime.now().strftime("%m%d-%H%M%S")
    filedir = 'train' if not (args.vol_output_only or args.render_only) else 'test'
    filedir += date_str
    logdir = os.path.join(expdir, filedir)
    os.makedirs(logdir, exist_ok=True)

    sys.stdout = Logger(logdir, False, fname="log.out")
    # sys.stderr = Logger(log_dir, False, fname="log.err")  # for tqdm

    print(" ".join(sys.argv), flush=True)
    printENV()

    # files backup
    shutil.copyfile(args.config, os.path.join(expdir, filedir, 'config.txt'))
    f = os.path.join(logdir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    filelist = ['run_pinf.py', 'run_pinf_helpers.py', 'pinf_rendering.py',
                # 'nerf/utils.py',
                # 'radiance_fields/nerf.py',
                'radiance_fields/siren.py',
                'radiance_fields/neus.py',
                ]
    if args.dataset_type == 'nv_data':
        filelist.append('datasets/neural_volumes.py')
    for filename in filelist:
        shutil.copyfile('./' + filename, os.path.join(logdir, filename.replace("/", "-")))

    return logdir


def create_model(name: str, args, input_ch: int = 3) -> RadianceField:
    D = args.netdepth
    W = args.netwidth
    if name == 'nerf':
        return NeRF(
            D=D, W=W, input_ch=input_ch, use_viewdirs=args.use_viewdirs,
            multires=args.multires, multires_views=args.multires_views
        )
    elif name == 'siren':
        return SIREN_NeRFt(
            D=D, W=W, input_ch=input_ch, use_viewdirs=args.use_viewdirs,
            first_omega_0=args.omega, unique_first=args.use_first_omega,
            fading_fin_step=args.fading_layers
        )
    elif name == 'neus':
        return NeuS(
            D=D, W=W, input_ch=input_ch,
            multires=args.multires, multires_views=args.multires_views,
            use_color_t=args.use_color_t,
            output_s_density=False
        )
    elif name == 'nsr':
        from radiance_fields.nsr import NSR
        return NSR(
            input_ch=input_ch,
            bound=2.5,
            fading_fin_step=20000,
        )
    elif name == 'hybrid':  # legacy PINF
        assert input_ch == 4
        static_model = SIREN_NeRFt(
            D=D, W=W, input_ch=3, use_viewdirs=args.use_viewdirs,
            first_omega_0=args.omega, unique_first=False,
            fading_fin_step=args.fading_layers
        )
        dynamic_model = SIREN_NeRFt(
            D=D, W=W, input_ch=4, use_viewdirs=args.use_viewdirs,
            first_omega_0=args.omega, unique_first=args.use_first_omega,
            fading_fin_step=args.fading_layers
        )
        return HybridRadianceField(static_model, dynamic_model)
    raise NotImplementedError(f"Unknown model name {name}")


def model_fading_update(model: RadianceField, prop_model: RadianceField | None, vel_model: SIREN_vel | None,
                        global_step, vel_delay):
    model.update_fading_step(global_step)
    if prop_model is not None:
        prop_model.update_fading_step(global_step)
    if vel_model is not None:
        vel_model.update_fading_step(global_step - vel_delay)


def load_model(ckpt_path: str,
               model: RadianceField, prop_model: RadianceField | None, optimizer: torch.optim.Optimizer,
               vel_model=None, vel_optimizer=None
               ) -> int:
    ckpt = torch.load(ckpt_path)

    # Load model
    model.load_state_dict(ckpt['network_state_dict'])
    if prop_model is not None:
        prop_model.load_state_dict(ckpt['network_prop_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # Load vel_model
    if vel_model is not None:
        if 'network_vel_state_dict' in ckpt:
            vel_model.load_state_dict(ckpt['network_vel_state_dict'])
        if 'vel_optimizer_state_dict' in ckpt:
            vel_optimizer.load_state_dict(ckpt['vel_optimizer_state_dict'])

    return ckpt['global_step']


def save_model(path: str, global_step: int,
               model: RadianceField, prop_model: RadianceField | None, optimizer: torch.optim.Optimizer,
               vel_model=None, vel_optimizer=None):
    save_dic = {
        'global_step': global_step,
        'network_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if prop_model is not None:
        save_dic['network_prop_state_dict'] = prop_model.state_dict()

    if vel_model is not None:
        save_dic['network_vel_state_dict'] = vel_model.state_dict()
        save_dic['vel_optimizer_state_dict'] = vel_optimizer.state_dict()

    torch.save(save_dic, path)


def convert_aabb(in_min, in_max, voxel_tran):
    in_min = torch.tensor(in_min, device=voxel_tran.device).expand(3)
    in_max = torch.tensor(in_max, device=voxel_tran.device).expand(3)
    in_min = pos_smoke2world(in_min, voxel_tran)
    in_max = pos_smoke2world(in_max, voxel_tran)
    cmp = torch.less(in_min, in_max)
    in_min, in_max = torch.where(cmp, in_min, in_max), torch.where(cmp, in_max, in_min)
    return torch.cat((in_min, in_max))


def pinf_train():
    args = config_parser().parse_args()
    set_rand_seed(args.fix_seed)

    # Create log dir and copy the config file
    expdir: str = os.path.join(args.basedir, args.expname)
    logdir: str = prepare_logging(expdir, args)
    writer = SummaryWriter(logdir=logdir)

    time0 = time.time()
    if args.dataset_type == 'pinf_data':
        pinf_data = PINFFrameData(args.datadir, half_res=args.half_res, normalize_time=True)

    elif args.dataset_type == 'nv_data':
        from datasets.neural_volumes import PINFFrameDataFromNV
        pinf_data = PINFFrameDataFromNV(args.datadir, is_test=args.render_only or args.vol_output_only,
                                        half_res=args.half_res)

    else:
        raise NotImplementedError(f"Unsupported dataset type {args.dataset_type}")

    train_data = PINFDataset(pinf_data)

    # used for in-training test-set
    test_data = PINFTestDataset(pinf_data, skip=args.testskip, video_id=0)

    print(f'Loading takes {time.time() - time0:.4f} s')
    time0 = time.time()

    video = train_data.videos[0]
    hwf = video.frames.shape[1], video.frames.shape[2], video.focal
    del video

    voxel_tran = pinf_data.voxel_tran
    voxel_scale = pinf_data.voxel_scale

    bkg_color = torch.Tensor(pinf_data.bkg_color).to(device)
    near, far = pinf_data.near, pinf_data.far
    if args.near > 0:
        near = args.near
    if args.far > 0:
        far = args.far
    t_info = pinf_data.t_info

    print(f'Conversion takes {time.time() - time0:.4f} s')

    # Load data
    # images, poses, time_steps, hwfs, render_poses, render_timesteps, i_split, t_info, voxel_tran, voxel_scale, bkg_color, near, far = load_pinf_frame_data(args.datadir, args.half_res, args.testskip)
    print('Loaded pinf frame data', args.datadir)
    print('Loaded voxel matrix', voxel_tran, 'voxel scale',  voxel_scale)

    voxel_tran[:3, :3] *= voxel_scale
    voxel_tran = torch.Tensor(voxel_tran).to(device)
    voxel_tran_inv = torch.inverse(voxel_tran)
    scene_scale = voxel_scale / voxel_scale[0]
    del voxel_scale

    print('Scene has background color', bkg_color)

    if args.render_test:
        render_data = test_data
    else:
        render_data = test_data.predictor()

    # Create Bbox model
    if args.bbox_min != "":
        in_min = [float(x) for x in args.bbox_min.split(",")]
        in_max = [float(x) for x in args.bbox_max.split(",")]
        aabb = convert_aabb(in_min, in_max, voxel_tran)
        print(f"aabb = {aabb}")
    else:
        aabb = None

    # Create vel model
    vel_model = None

    if args.nseW > 1e-8:
        # D=6, W=128, input_ch=4, output_ch=3, skips=[],
        vel_model = SIREN_vel(fading_fin_step=args.fading_layers).to(device)

    # Create nerf model
    input_ch = 4

    model = create_model(args.net_model, args, input_ch).to(device)
    grad_vars = list(model.parameters())
    if args.s_model != '':
        model_s = create_model(args.s_model, args, input_ch - 1).to(device)
        grad_vars += model_s.parameters()
        model = HybridRadianceField(model_s, model)

    prop_model = None
    if args.N_importance > 0:
        prop_model = create_model(args.net_model, args, input_ch).to(device)
        grad_vars += list(prop_model.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    vel_optimizer = None
    if vel_model is not None:
        vel_grad_vars = list(vel_model.parameters())
        vel_optimizer = torch.optim.Adam(params=vel_grad_vars, lr=args.lrate, betas=(0.9, 0.999))

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
        start = load_model(ckpt_path, model, prop_model, optimizer, vel_model, vel_optimizer)
    ##########################

    renderer = PINFRenderer(
        model=model,
        prop_model=prop_model,
        n_samples=args.N_samples,
        n_importance=args.N_importance,
        near=near,
        far=far,
        perturb=args.perturb > 0,
        vel_model=vel_model,
        aabb=aabb,
    )

    global_step = start

    velInStep = max(0,args.vel_delay) if args.nseW > 1e-8 else 0 # after tempoInStep
    model_fading_update(model, prop_model, vel_model, start, velInStep)

    test_bkg_color = torch.tensor([0.0, 0.0, 0.3], device=device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')

        from lpips import LPIPS
        lpips_net = LPIPS(net="vgg").to(device)
        lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
        lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

        if args.render_test:
            test_bkg_color = bkg_color

        with torch.no_grad():
            testsavedir = os.path.join(expdir, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start+1))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', len(render_data))

            if isinstance(model, HybridRadianceField) and isinstance(model.static_model, SDFRadianceField):
                from neus_utils import extract_geometry
                import trimesh

                mesh_path = os.path.join(testsavedir, '{:0>8d}.ply'.format(start))
                if not os.path.exists(mesh_path):
                    print('Exporting mesh from sdf')
                    sdf_model = model.static_model
                    vertices, triangles = extract_geometry(aabb[:3], aabb[3:], 256, 0.0,
                                                           lambda pts: -sdf_model.sdf(pts.to(device)))
                    mesh = trimesh.Trimesh(vertices, triangles)
                    mesh.export(mesh_path)
                    del vertices, triangles, mesh

            render_path(renderer, render_data, hwf, args.chunk, save_dir=testsavedir,
                        video_prefix=os.path.join(testsavedir, 'video_'),
                        render_factor=args.render_factor, bkgd_color=test_bkg_color, lpips_fn=lpips_fn)

            return

    if args.vol_output_only:
        print('OUTPUT VOLUME ONLY')
        savenpz = True # need a large space
        savejpg = True
        save_vort = True # (vel_model is not None) and (savenpz) and (savejpg)
        with torch.no_grad():
            testsavedir = os.path.join(expdir, 'volumeout_{:06d}'.format(start+1))
            os.makedirs(testsavedir, exist_ok=True)

            voxel_writer = VoxelTool(voxel_tran, voxel_tran_inv, scene_scale, args.vol_output_W)

            # t_list = list(np.arange(t_info[0],t_info[1],t_info[-1]))
            # frame_N = len(t_list)
            # noStatic = False
            # frames = np.arange(frame_N//10, frame_N, 10)
            # frames = np.sort(np.concatenate((frames, frames + 1)))
            # for frame_i in frames:
            #     print(frame_i, frame_N)
            #     cur_t = t_list[frame_i]
            #     voxel_writer.save_voxel_den_npz(
            #         os.path.join(testsavedir,"d_%04d.npz"%frame_i), cur_t,
            #         network_fn=model,
            #         chunk=args.chunk, save_npz=savenpz, save_jpg=savejpg, noStatic=noStatic)
            #     noStatic = True
            #     if vel_model is not None:
            #         voxel_writer.save_voxel_vel_npz(os.path.join(testsavedir,"v_%04d.npz"%frame_i), t_info[-1], cur_t, args.chunk, vel_model, savenpz, savejpg, save_vort)
            # print('Done output', testsavedir)

            v_deltaT = 0.025
            with torch.no_grad():
                vel_rgbs = []
                for _t in trange(int(1.0 / v_deltaT)):
                    # middle_slice, True: only sample middle slices for visualization, very fast, but cannot save as npz
                    #               False: sample whole volume, can be saved as npz, but very slow

                    voxel_den_list = voxel_writer.get_voxel_density_list(
                        t=_t * v_deltaT, network_fn=model, middle_slice=False, opaque=True)[::-1]
                    smoke_den = voxel_den_list[-1]
                    voxel_den_list.append(voxel_writer.get_voxel_velocity(
                        t_info[-1], _t * v_deltaT, args.chunk, vel_model, middle_slice=True, ref_den_list=smoke_den))
                    voxel_img = []
                    for voxel in voxel_den_list:
                        voxel = voxel.detach().cpu().numpy()
                        if voxel.shape[-1] == 1:
                            voxel_img.append(
                                np.repeat(den_scalar2rgb(voxel, scale=None, is3D=True, logv=False, mix=True), 3,
                                          axis=-1))
                        else:
                            voxel_img.append(vel_uv2hsv(voxel, scale=300, is3D=True, logv=False))
                    voxel_img = np.concatenate(voxel_img, axis=0)  # 128,64*3,3

                    vel_rgbs.append(voxel_img)
            moviebase = os.path.join(testsavedir, 'velrgb.mp4')
            imageio.mimwrite(moviebase, np.stack(vel_rgbs, axis=0).astype(np.uint8), fps=30, quality=8)

            return

    n_rand = args.N_rand

    # Prepare Loss Tools (VGG, Den2Vel)
    ###############################################
    vgg_tool = VGGLossTool(device)

    # Move to GPU, except images
    # poses = torch.Tensor(poses).to(device)
    # timesteps = torch.Tensor(timesteps).to(device)

    n_iters = args.N_iter + 1

    print('Begin')

    # Prepare Voxel Sampling Tools for Image Summary (voxel_writer), Physical Priors (training_voxel), Data Priors Represented by D2V (den_p_all)
    # voxel_writer: to sample low resolution data for for image summary
    # voxel_writer = VoxelTool(voxel_tran, voxel_tran_inv, scene_scale, 64)

    # training_voxel: to sample data for for velocity NSE training
    # training_voxel should have a larger resolution than voxel_writer
    # note that training voxel is also used for visualization in testing
    min_ratio = float(64+4*2)/min(scene_scale[0], scene_scale[1], scene_scale[2])
    train_x = max(args.vol_output_W, int(min_ratio * scene_scale[0] + 0.5))
    training_voxel = VoxelTool(voxel_tran, voxel_tran_inv, scene_scale, train_x)
    training_pts = torch.reshape(training_voxel.pts, (-1,3))
    voxel_writer = training_voxel

    split_nse_wei = [2.0, 1e-3, 1e-3, 1e-3, 5e-3, 5e-3]  # den, vel*3, div, scale
    start = start + 1

    testimgdir = logdir + "_imgs"
    os.makedirs(testimgdir, exist_ok=True)
    # some loss terms 
    ghost_loss, overlay_loss, nseloss_fine = None, None, None
    time0 = time.time()
    psnr1k = np.zeros(1000)

    for i in trange(start, n_iters):
        model_fading_update(model, prop_model, vel_model, global_step, velInStep)

        # train radiance all the time, train vel less, train with d2v even less.
        trainVGG = (args.vggW > 0.0) and (i % 4 == 0) # less vgg training
        trainVel = (global_step >= velInStep) and (vel_model is not None) and (i % 10 == 0)
        neumann_loss = None

        # fading in for networks
        tempo_fading = fade_in_weight(global_step, 0, args.tempo_fading)
        vel_fading = fade_in_weight(global_step, velInStep, 10000)
        warp_fading = fade_in_weight(global_step, velInStep+10000, 20000)
        # fading in for losses
        vgg_fading = [fade_in_weight(global_step, (vgg_i-1)*10000, 10000) for vgg_i in range(len(vgg_tool.layer_list),0,-1)]
        ghost_fading = fade_in_weight(global_step, 2000, 20000)
        ###########################################################

        # Random from one frame
        video, frame_i = train_data.get_video_and_frame(np.random.randint(len(train_data)))
        target = torch.Tensor(video.frames[frame_i]).to(device)
        K = video.intrinsics()
        H, W = target.shape[:2]
        pose = torch.Tensor(video.c2w(frame_i)).to(device)
        time_locate = t_info[-1] * frame_i
        if hasattr(video, 'background'):
            background = torch.tensor(video.background, device=device)
        else:
            background = bkg_color

        if trainVel:
            # take a mini_batch 32*32*32
            train_x, train_y, train_z = training_voxel.voxel_size()
            train_random = np.random.choice(train_z*train_y*train_x, 32*32*32)
            training_samples = training_pts[train_random]

            training_samples = training_samples.view(-1,3)
            training_t = torch.ones([training_samples.shape[0], 1], device=device)*time_locate
            training_samples = torch.cat([training_samples,training_t], dim=-1)

            #####  core velocity optimization loop  #####
            # allows to take derivative w.r.t. training_samples
            training_samples = training_samples.detach().requires_grad_(True)
            _vel, _u_x, _u_y, _u_z, _u_t = get_velocity_and_derivatives(training_samples, chunk=args.chunk, vel_model=vel_model)
            if args.vel_no_slip:
                smoke_model = model
            else:
                smoke_model = model.dynamic_model if isinstance(model, HybridRadianceField) else model
            _den, _d_x, _d_y, _d_z, _d_t = get_density_and_derivatives(
                training_samples, chunk=args.chunk,
                network_fn=smoke_model,
                opaque=True,    # for neus
            )

            vel_optimizer.zero_grad()
            split_nse = PDE_EQs(
                _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
                _vel, _u_t, _u_x, _u_y, _u_z)
            nse_errors = [torch.mean(torch.square(x)) for x in split_nse]
            nseloss_fine = 0.0
            for ei,wi in zip (nse_errors, split_nse_wei):
                nseloss_fine = ei*wi + nseloss_fine
            vel_loss = nseloss_fine * args.nseW * vel_fading

            # Neumann loss
            if isinstance(model, HybridRadianceField) and isinstance(model.static_model, SDFRadianceField):
                sdf_model = model.static_model
                with torch.no_grad():
                    if isinstance(sdf_model, NeuS):
                        sdf, gradient = sdf_model.forward_with_gradient(training_samples[..., :3])
                        sdf = sdf[..., :1]
                    else:
                        sdf = sdf_model.sdf(training_samples[..., :3])
                        gradient = sdf_model.gradient(training_samples[..., :3])
                sdf, gradient = sdf.detach(), gradient.detach()
                neumann_loss = sdf_model.opaque_density(sdf).detach() * F.relu(-torch.sum(_vel * gradient, dim=-1, keepdim=True))
                neumann_loss = torch.mean(neumann_loss)
                if args.neumann > 0.0:
                    vel_loss = vel_loss + neumann_loss * args.neumann
                del sdf, gradient
                neumann_loss = neumann_loss.detach()
                # writer.add_scalar('Neumann loss', neumann_loss, i)

            vel_loss.backward()
            vel_optimizer.step()

            # cleanup
            del _vel, _u_x, _u_y, _u_z, _u_t, _den, _d_x, _d_y, _d_z, _d_t, split_nse
            nse_errors = tuple(x.item() for x in nse_errors)
            nseloss_fine = nseloss_fine.item()
            vel_loss = vel_loss.item()

        if trainVGG: # get a cropped img (dw,dw) to train vgg
            coords_crop, dw = vgg_sample(args.vgg_strides, n_rand, target, bkg_color, steps=i)
            coords_crop = torch.reshape(coords_crop, [-1, 2])
            ys, xs = coords_crop[:, 0], coords_crop[:, 1]  # vgg_sample using ij, convert to xy
        else:
            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                xs, ys = torch.meshgrid(
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    indexing='xy'
                )
                selected = np.random.choice(4 * dH * dW, size=[n_rand], replace=False)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
            else:
                xs, ys = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
                selected = np.random.choice(H * W, size=[n_rand], replace=False)
            xs = torch.flatten(xs)[selected].to(device)
            ys = torch.flatten(ys)[selected].to(device)

        rays = get_rays(K, pose, xs, ys)  # (n_rand, 3), (n_rand, 3)
        rays = rays.foreach(lambda t: t.to(device))
        target_s = target[ys.long(), xs.long()]  # (n_rand, 3)

        if background is not None and background.dim() > 2:
            background = background[ys.long(), xs.long()]

        if args.train_warp and vel_model is not None and (global_step >= velInStep):
            renderer.warp_fading_dt = warp_fading * t_info[-1]
            # fading * delt_T, need to update every iteration
        # renderer.cos_anneal_ratio = min(i / 50000, 1.0)  # for neus

        #####  core radiance optimization loop  #####
        output = renderer.render(
            rays.origins, rays.viewdirs, chunk=args.chunk,
            ret_raw=True,
            timestep=time_locate,
            background=background)
        rgb, _, acc, extras = output.as_tuple()
        out0: NeRFOutputs | None = extras.get('coarse')

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        if 'static' in extras and tempo_fading < 1.0 - 1e-8:
            img_loss = img_loss * tempo_fading + img2mse(extras['static'].rgb, target_s) * (1.0-tempo_fading)
            # rgb = rgb * tempo_fading + extras['rgbh1'] * (1.0-tempo_fading)

        # trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss.detach())
        psnr1k[i % 1000] = psnr
        # writer.add_scalar('PSNR', psnr, i)

        if out0 is not None:
            img_loss0 = img2mse(out0.rgb, target_s)
            if 'static' in out0.extras and tempo_fading < 1.0 - 1e-8:
                img_loss0 = img_loss0 * tempo_fading + img2mse(out0.extras['static'].rgb, target_s) * (1.0-tempo_fading)
                # extras['rgb0'] = extras['rgb0'] * tempo_fading + extras['rgbh10'] * (1.0-tempo_fading)

            loss = loss + img_loss0
            # psnr0 = mse2psnr(img_loss0)

        if trainVGG:
            vgg_loss_func = vgg_tool.compute_cos_loss
            vgg_tar = torch.reshape(target_s, [dw,dw,3])
            vgg_img = torch.reshape(rgb, [dw,dw,3])
            vgg_loss = vgg_loss_func(vgg_img, vgg_tar)
            w_vgg = args.vggW / float(len(vgg_loss))
            vgg_loss_sum = 0
            for _w, _wf in zip(vgg_loss, vgg_fading):
                if _wf > 1e-8:
                    vgg_loss_sum = _w * _wf * w_vgg + vgg_loss_sum

            if out0 is not None:
                vgg_img0 = torch.reshape(out0.rgb, [dw,dw,3])
                vgg_loss0 = vgg_loss_func(vgg_img0, vgg_tar)
                for _w, _wf in zip(vgg_loss0, vgg_fading):
                    if _wf > 1e-8:
                        vgg_loss_sum = _w * _wf * w_vgg + vgg_loss_sum
            loss += vgg_loss_sum

        if (args.ghostW > 0.0) and background is not None:
            w_ghost = ghost_fading * args.ghostW
            ghost_scale = args.ghost_scale
            if w_ghost > 1e-8:
                ghost_loss = ghost_loss_func(output, background, ghost_scale)
                if 'static' in extras:# static part
                    ghost_loss += 0.1 * ghost_loss_func(extras['static'], background, ghost_scale)
                    if 'dynamic' in extras:# dynamic part
                        ghost_loss += 0.1 * ghost_loss_func(extras['dynamic'], extras['static'].rgb, ghost_scale)

                if out0 is not None:
                    ghost_loss0 = ghost_loss_func(out0, background, ghost_scale)
                    if 'static' in out0.extras:  # static part
                        # ghost_loss0 += 0.1*ghost_loss_func(extras['rgbh10'], static_back, extras['acch10'], den_penalty=0.0)
                        if 'dynamic' in out0.extras:  # dynamic part
                            ghost_loss += 0.1 * ghost_loss_func(out0.extras['dynamic'], out0.extras['static'].rgb,
                                                                ghost_scale)

                    ghost_loss += ghost_loss0

                loss += ghost_loss * w_ghost

        w_overlay = args.overlayW * ghost_fading  # with fading
        if 'static' in extras and w_overlay > 0:
            # density should be either from smoke or from static, not mixed.
            smoke_den = extras['sigma_d']
            if 'sdf' in extras:
                inv_s = extras['inv_s'].detach()  # as constant
                # static_den = inv_s * torch.tanh(F.relu(-inv_s * extras['sdf']))  # tanh(x) = 2 * sigmoid(2x) - 1
                static_den = inv_s * torch.sigmoid(-inv_s * extras['sdf']) / 2  # opaque_density
                # static_den = extras['sigma_s'].detach()
                writer.add_scalar('inv_s', inv_s, i)
            else:
                static_den = extras['sigma_s']
            overlay_loss = (smoke_den * static_den) / (torch.square(smoke_den) + torch.square(static_den) + 1e-8)
            overlay_loss = torch.mean(overlay_loss)
            loss += overlay_loss * w_overlay

        eikonal_loss = extras.get('eikonal_loss')
        eikonal_weight = args.eikonal
        if eikonal_loss is not None and eikonal_weight > 0:
            # eikonal_weight = np.clip(i / 10000, 0.0, 1.0) * 0.001
            loss += eikonal_loss * eikonal_weight
            # writer.add_scalar('eikonal_loss', eikonal_loss, i)
            if i > 20000 and extras['inv_s'] < 100.0 and args.devW > 0:
                loss += args.devW / extras['inv_s']

        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        if trainVel and vel_optimizer is not None:
            for param_group in vel_optimizer.param_groups:
                param_group['lr'] = new_lrate

        ################################
        # dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        if i == 10 or i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  avg. {psnr1k.mean()}")
            print(f"Average speed {time.time() - time0:.4f} s / {args.i_print} steps")

            if args.N_importance > 0:
                print("img_loss: ", img_loss.item(), img_loss0.item())
            else:
                print("img_loss: ", img_loss.item())

            if trainVGG:
                print("vgg_loss: %0.4f *" % w_vgg, vgg_loss_sum.item())
                for vgg_i in range(len(vgg_loss)):
                    _wf = vgg_fading[vgg_i]
                    if args.N_importance > 0:
                        print(vgg_tool.layer_names[vgg_i], vgg_loss[vgg_i].item(), "+", vgg_loss0[vgg_i].item(),
                              "with vgg_fading: %0.4f" % _wf)
                    else:
                        print(vgg_tool.layer_names[vgg_i], vgg_loss[vgg_i].item(), "with vgg_fading: %0.4f" % _wf)

            if ghost_loss is not None:
                _g = ghost_loss.item()
                _cg = ghost_loss0.item() if args.N_importance > 0 else 0
                print("w_ghost: %0.4f," % w_ghost, "ghost_loss: ", _g, "coarse: ", _cg, "fine: ", _g - _cg)

            if overlay_loss is not None:
                print("w_overlay: %0.4f," % w_overlay, "overlay_loss: ", overlay_loss.item())

            if eikonal_loss is not None:
                print(f"eikonal loss: {eikonal_loss.item():.6f}  inv_s: {extras.get('inv_s')}")

            if trainVel:
                print("vel_loss: ", vel_loss)
                print("neumann_loss: ", neumann_loss)

                if nseloss_fine is not None:
                    print(" ".join(["nse(e1-e6):"] + [str(ei) for ei in nse_errors]))
                    print("NSE loss sum = ", nseloss_fine,
                          "* w_nse(%0.4f) * vel_fading(%0.4f)" % (args.nseW, vel_fading))

            for _m in (model, prop_model, vel_model):
                if _m is not None:
                    _m.print_fading()

            sys.stdout.flush()

        if i in (1, 100, 400, 1000, 20010) or i % args.i_img == 0:
            with torch.no_grad():
                if trainVGG:
                    vgg_img = np.concatenate([vgg_tar.cpu().detach().numpy(), vgg_img.cpu().detach().numpy()], axis=1)
                    imageio.imwrite(os.path.join(testimgdir, 'vggcmp_{:06d}.png'.format(i)), to8b(vgg_img))

                voxel_output(voxel_writer, model, None,
                             vel_model if trainVel else None, testimgdir,
                             step=i, timestep=0.5, scale=None)

        # Rest is logging
        if (i in (10000, 20000, 40000) or i % args.i_weights == 0) and i > start + 1:
            path = os.path.join(expdir, '{:06d}.tar'.format(i))
            save_model(path, global_step, model, prop_model, optimizer, vel_model, vel_optimizer)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0 and i > start + 1:
            testsavedir = os.path.join(expdir, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses', len(test_data))
            with torch.no_grad():
                render_path(renderer, test_data, hwf, args.chunk, save_dir=testsavedir, bkgd_color=bkg_color)
            print('Saved test set')

        if i % args.i_video == 0 and i > start + 1:
            # Turn on testing mode
            moviebase = os.path.join(expdir, 'spiral_{:06d}_'.format(i))
            with torch.no_grad():
                # the path rendering can be very slow.
                render_path(renderer, render_data, hwf, args.chunk, bkgd_color=test_bkg_color,
                            video_prefix=moviebase)

            v_deltaT = 0.025
            with torch.no_grad():
                vel_rgbs = []
                for _t in range(int(1.0 / v_deltaT)):
                    # middle_slice, True: only sample middle slices for visualization, very fast, but cannot save as npz
                    #               False: sample whole volume, can be saved as npz, but very slow
                    voxel_vel = training_voxel.get_voxel_velocity(t_info[-1], _t * v_deltaT, args.chunk, vel_model,
                                                                  middle_slice=True)
                    voxel_vel = voxel_vel.view([-1] + list(voxel_vel.shape))
                    _, voxel_vort = jacobian3D(voxel_vel)
                    _vel = vel_uv2hsv(np.squeeze(voxel_vel.detach().cpu().numpy()), scale=300, is3D=True, logv=False)
                    _vort = vel_uv2hsv(np.squeeze(voxel_vort.detach().cpu().numpy()), scale=1500, is3D=True, logv=False)
                    vel_rgbs.append(np.concatenate([_vel, _vort], axis=0))
            moviebase = os.path.join(expdir, 'volume_{:06d}_'.format(i))
            imageio.mimwrite(moviebase + 'velrgb.mp4', np.stack(vel_rgbs, axis=0).astype(np.uint8), fps=30, quality=8)

        global_step += 1
        if i % args.i_print == 0:
            time0 = time.time()  # reset timer


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if sys.gettrace() is not None:  # in debug mode
        pinf_train()
    else:
        try:
            pinf_train()
        except Exception as e:
            import traceback
            traceback.print_exception(e, file=sys.stdout)  # write to log
