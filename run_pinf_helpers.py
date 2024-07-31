import sys
import os

import numpy as np
import imageio
# torch.autograd.set_detect_anomaly(True)
import torch
from torch.nn import functional as F
import cv2 as cv

from nerf.utils import NeRFOutputs, attach_time
from radiance_fields import RadianceField, HybridRadianceField


#####################################################################
# custom Logger to write Log to file
class Logger(object):
    def __init__(self, summary_dir, silent=False, fname="logfile.txt"):
        self.terminal = sys.stdout
        self.silent = silent
        self.log = open(os.path.join(summary_dir, fname), "a") 
        cmdline = " ".join(sys.argv)+"\n"
        self.log.write(cmdline) 
    def write(self, message):
        if not self.silent: 
            self.terminal.write(message)
        self.log.write(message) 
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def printENV():
    check_list = ['CUDA_VISIBLE_DEVICES']
    for name in check_list:
        if name in os.environ:
            print(name, os.environ[name])
        else:
            print(name, "Not find")

    sys.stdout.flush()


#####################################################################
# Visualization Tools

def velLegendHSV(hsvin, is3D, lw=-1, constV=255):
    # hsvin: (b), h, w, 3
    # always overwrite hsvin borders [lw], please pad hsvin before hand
    # or fill whole hsvin (lw < 0)
    ih, iw = hsvin.shape[-3:-1]
    if lw<=0: # fill whole
        a_list, b_list = [range(ih)], [range(iw)]
    else: # fill border
        a_list = [range(ih),  range(lw), range(ih), range(ih-lw, ih)]
        b_list = [range(lw),  range(iw), range(iw-lw, iw), range(iw)]
    for a,b in zip(a_list, b_list):
        for _fty in a:
            for _ftx in b:
                fty = _fty - ih//2
                ftx = _ftx - iw//2
                ftang = np.arctan2(fty, ftx) + np.pi
                ftang = ftang*(180/np.pi/2)
                # print("ftang,min,max,mean", ftang.min(), ftang.max(), ftang.mean())
                # ftang,min,max,mean 0.7031249999999849 180.0 90.3515625
                hsvin[...,_fty,_ftx,0] = np.expand_dims(ftang, axis=-1) # 0-360 
                # hsvin[...,_fty,_ftx,0] = ftang
                hsvin[...,_fty,_ftx,2] = constV
                if (not is3D) or (lw == 1):
                    hsvin[...,_fty,_ftx,1] = 255
                else:
                    thetaY1 = 1.0 - ((ih//2) - abs(fty)) / float( lw if (lw > 1) else (ih//2) )
                    thetaY2 = 1.0 - ((iw//2) - abs(ftx)) / float( lw if (lw > 1) else (iw//2) )
                    fthetaY = max(thetaY1, thetaY2) * (0.5*np.pi)
                    ftxY, ftyY = np.cos(fthetaY), np.sin(fthetaY)
                    fangY = np.arctan2(ftyY, ftxY)
                    fangY = fangY*(240/np.pi*2) # 240 - 0
                    hsvin[...,_fty,_ftx,1] = 255 - fangY
                    # print("fangY,min,max,mean", fangY.min(), fangY.max(), fangY.mean())
    # finished velLegendHSV.

def cubecenter(cube, axis, half = 0):
    # cube: (b,)h,h,h,c
    # axis: 1 (z), 2 (y), 3 (x)
    reduce_axis = [a for a in [1,2,3] if a != axis]
    pack = np.mean(cube, axis=tuple(reduce_axis)) # (b,)h,c
    pack = np.sqrt(np.sum( np.square(pack), axis=-1 ) + 1e-6) # (b,)h

    length = cube.shape[axis-5] # h
    weights = np.arange(0.5/length,1.0,1.0/length)
    if half == 1: # first half
        weights = np.where( weights < 0.5, weights, np.zeros_like(weights))
        pack = np.where( weights < 0.5, pack, np.zeros_like(pack))
    elif half == 2: # second half
        weights = np.where( weights > 0.5, weights, np.zeros_like(weights))
        pack = np.where( weights > 0.5, pack, np.zeros_like(pack))

    weighted = pack * weights # (b,)h
    weiAxis = np.sum(weighted, axis=-1) / np.sum(pack, axis=-1) * length # (b,)
    
    return weiAxis.astype(np.int32) # a ceiling is included

def vel2hsv(velin, is3D, logv, scale=None): # 2D
    fx, fy = velin[...,0], velin[...,1]
    ori_shape = list(velin.shape[:-1]) + [3]
    if is3D: 
        fz = velin[...,2]
        ang = np.arctan2(fz, fx) + np.pi # angXZ
        zxlen2 = fx*fx+fz*fz
        angY = np.arctan2(np.abs(fy), np.sqrt(zxlen2))
        v = np.sqrt(zxlen2+fy*fy)
    else:
        v = np.sqrt(fx*fx+fy*fy)
        ang = np.arctan2(fy, fx) + np.pi
    
    if logv:
        v = np.log10(v+1)
    
    hsv = np.zeros(ori_shape, np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    if is3D:
        hsv[...,1] = 255 - angY*(240/np.pi*2)  
    else:
        hsv[...,1] = 255
    if scale is not None:
        hsv[...,2] = np.minimum(v*scale, 255)
    else:
        hsv[...,2] = v/max(v.max(),1e-6) * 255.0
    return hsv


def vel_uv2hsv(vel, scale = 160, is3D=False, logv=False, mix=False):
    # vel: a np.float32 array, in shape of (?=b,) d,h,w,3 for 3D and (?=b,)h,w, 2 or 3 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good. 
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use more slices to get a volumetric visualization if True, which is slow

    ori_shape = list(vel.shape[:-1]) + [3] # (?=b,) d,h,w,3
    if is3D: 
        new_range = list( range( len(ori_shape) ) )
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXvel = np.transpose(vel, z_new_range)
        
        _xm,_ym,_zm = (ori_shape[-2]-1)//2, (ori_shape[-3]-1)//2, (ori_shape[-4]-1)//2
        
        if mix:
            _xlist = [cubecenter(vel, 3, 1),_xm,cubecenter(vel, 3, 2)]
            _ylist = [cubecenter(vel, 2, 1),_ym,cubecenter(vel, 2, 2)]
            _zlist = [cubecenter(vel, 1, 1),_zm,cubecenter(vel, 1, 2)]
        else:
            _xlist, _ylist, _zlist = [_xm], [_ym], [_zm]

        hsv = []
        for _x, _y, _z in zip (_xlist, _ylist, _zlist):
            # print(_x, _y, _z)
            _x, _y, _z = np.clip([_x, _y, _z], 0, ori_shape[-2:-5:-1])
            _yz = YZXvel[...,_x,:]
            _yz = np.stack( [_yz[...,2],_yz[...,0],_yz[...,1]], axis=-1)
            _yx = YZXvel[...,_z,:,:]
            _yx = np.stack( [_yx[...,0],_yx[...,2],_yx[...,1]], axis=-1)
            _zx = YZXvel[...,_y,:,:,:]
            _zx = np.stack( [_zx[...,0],_zx[...,1],_zx[...,2]], axis=-1)
            # print(_yx.shape, _yz.shape, _zx.shape)

            # in case resolution is not a cube, (res,res,res)
            _yxz = np.concatenate( [ #yz, yx, zx
                _yx, _yz ], axis = -2) # (?=b,),h,w+zdim,3
            
            if ori_shape[-3] < ori_shape[-4]:
                pad_shape = list(_yxz.shape) #(?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
                _pad = np.zeros(pad_shape, dtype=np.float32)
                _yxz = np.concatenate( [_yxz,_pad], axis = -3)
            elif ori_shape[-3] > ori_shape[-4]:
                pad_shape = list(_zx.shape) #(?=b,),h,w+zdim,3
                pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

                _zx = np.concatenate( 
                    [_zx,np.zeros(pad_shape, dtype=np.float32)], axis = -3)
            
            midVel = np.concatenate( [ #yz, yx, zx
                _yxz, _zx
            ], axis = -2) # (?=b,),h,w*3,3
            hsv += [vel2hsv(midVel, True, logv, scale)]
        # remove depth dim, increase with zyx slices
        ori_shape[-3] = 3 * ori_shape[-2]
        ori_shape[-2] = ori_shape[-1]
        ori_shape = ori_shape[:-1]
    else:
        hsv = [vel2hsv(vel, False, logv, scale)]

    bgr = []
    for _hsv in hsv:
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape([-1]+ori_shape[-2:])
        if is3D:
            velLegendHSV(_hsv, is3D, lw=max(1,min(6,int(0.025*ori_shape[-2]))), constV=255)
        _hsv = cv.cvtColor(_hsv, cv.COLOR_HSV2BGR)
        if len(ori_shape) > 3:
            _hsv = _hsv.reshape(ori_shape)
        bgr += [_hsv]
    if len(bgr) == 1:
        bgr = bgr[0]
    else:
        bgr = bgr[0] * 0.2 + bgr[1] * 0.6 + bgr[2] * 0.2
    return bgr.astype(np.uint8)[::-1] # flip Y


def den_scalar2rgb(den, scale: float | None = 160.0, is3D=False, logv=False, mix=True):
    # den: a np.float32 array, in shape of (?=b,) d,h,w,1 for 3D and (?=b,)h,w,1 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good. 
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use averaged value as a volumetric visualization if True, else show middle slice

    ori_shape = list(den.shape)
    if ori_shape[-1] != 1:
        ori_shape.append(1)
        den = np.reshape(den, ori_shape)

    if is3D: 
        new_range = list( range( len(ori_shape) ) )
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXden = np.transpose(den, z_new_range)
                
        if not mix:
            _yz = YZXden[...,(ori_shape[-2]-1)//2,:]
            _yx = YZXden[...,(ori_shape[-4]-1)//2,:,:]
            _zx = YZXden[...,(ori_shape[-3]-1)//2,:,:,:]
        else:
            _yz = np.average(YZXden, axis=-2)
            _yx = np.average(YZXden, axis=-3)
            _zx = np.average(YZXden, axis=-4)
            # print(_yx.shape, _yz.shape, _zx.shape)

        # in case resolution is not a cube, (res,res,res)
        _yxz = np.concatenate( [ #yz, yx, zx
            _yx, _yz ], axis = -2) # (?=b,),h,w+zdim,1
        
        if ori_shape[-3] < ori_shape[-4]:
            pad_shape = list(_yxz.shape) #(?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
            _pad = np.zeros(pad_shape, dtype=np.float32)
            _yxz = np.concatenate( [_yxz,_pad], axis = -3)
        elif ori_shape[-3] > ori_shape[-4]:
            pad_shape = list(_zx.shape) #(?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

            _zx = np.concatenate( 
                [_zx,np.zeros(pad_shape, dtype=np.float32)], axis = -3)
        
        midDen = np.concatenate( [ #yz, yx, zx
            _yxz, _zx
        ], axis = -2) # (?=b,),h,w*3,1
    else:
        midDen = den

    if logv:
        midDen = np.log10(midDen+1)
    if scale is None:
        midDen = midDen / max(midDen.max(),1e-6) * 255.0
    else:
        midDen = midDen * scale
    grey = np.clip(midDen, 0, 255)

    return grey.astype(np.uint8)[::-1] # flip y


#####################################################################
# Physics Tools

def jacobian3D(x):
    # x, (b,)d,h,w,ch, pytorch tensor
    # return jacobian and curl

    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = torch.cat((dudx, torch.unsqueeze(dudx[:,:,:,-1], 3)), 3)
    dvdx = torch.cat((dvdx, torch.unsqueeze(dvdx[:,:,:,-1], 3)), 3)
    dwdx = torch.cat((dwdx, torch.unsqueeze(dwdx[:,:,:,-1], 3)), 3)

    dudy = torch.cat((dudy, torch.unsqueeze(dudy[:,:,-1,:], 2)), 2)
    dvdy = torch.cat((dvdy, torch.unsqueeze(dvdy[:,:,-1,:], 2)), 2)
    dwdy = torch.cat((dwdy, torch.unsqueeze(dwdy[:,:,-1,:], 2)), 2)

    dudz = torch.cat((dudz, torch.unsqueeze(dudz[:,-1,:,:], 1)), 1)
    dvdz = torch.cat((dvdz, torch.unsqueeze(dvdz[:,-1,:,:], 1)), 1)
    dwdz = torch.cat((dwdz, torch.unsqueeze(dwdz[:,-1,:,:], 1)), 1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = torch.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], -1)
    c = torch.stack([u,v,w], -1)
    
    return j, c


def jacobian3D_np(x):
    # x, (b,)d,h,w,ch
    # return jacobian and curl

    if len(x.shape) < 5:
        x = np.expand_dims(x, axis=0)
    dudx = x[:,:,:,1:,0] - x[:,:,:,:-1,0]
    dvdx = x[:,:,:,1:,1] - x[:,:,:,:-1,1]
    dwdx = x[:,:,:,1:,2] - x[:,:,:,:-1,2]
    dudy = x[:,:,1:,:,0] - x[:,:,:-1,:,0]
    dvdy = x[:,:,1:,:,1] - x[:,:,:-1,:,1]
    dwdy = x[:,:,1:,:,2] - x[:,:,:-1,:,2]
    dudz = x[:,1:,:,:,0] - x[:,:-1,:,:,0]
    dvdz = x[:,1:,:,:,1] - x[:,:-1,:,:,1]
    dwdz = x[:,1:,:,:,2] - x[:,:-1,:,:,2]

    # u = dwdy[:,:-1,:,:-1] - dvdz[:,:,1:,:-1]
    # v = dudz[:,:,1:,:-1] - dwdx[:,:-1,1:,:]
    # w = dvdx[:,:-1,1:,:] - dudy[:,:-1,:,:-1]

    dudx = np.concatenate((dudx, np.expand_dims(dudx[:,:,:,-1], axis=3)), axis=3)
    dvdx = np.concatenate((dvdx, np.expand_dims(dvdx[:,:,:,-1], axis=3)), axis=3)
    dwdx = np.concatenate((dwdx, np.expand_dims(dwdx[:,:,:,-1], axis=3)), axis=3)

    dudy = np.concatenate((dudy, np.expand_dims(dudy[:,:,-1,:], axis=2)), axis=2)
    dvdy = np.concatenate((dvdy, np.expand_dims(dvdy[:,:,-1,:], axis=2)), axis=2)
    dwdy = np.concatenate((dwdy, np.expand_dims(dwdy[:,:,-1,:], axis=2)), axis=2)

    dudz = np.concatenate((dudz, np.expand_dims(dudz[:,-1,:,:], axis=1)), axis=1)
    dvdz = np.concatenate((dvdz, np.expand_dims(dvdz[:,-1,:,:], axis=1)), axis=1)
    dwdz = np.concatenate((dwdz, np.expand_dims(dwdz[:,-1,:,:], axis=1)), axis=1)

    u = dwdy - dvdz
    v = dudz - dwdx
    w = dvdx - dudy
    
    j = np.stack([dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz], axis=-1)
    c = np.stack([u,v,w], axis=-1)
    
    return j, c


# from FFJORD github code
def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac

# from FFJORD github code
def divergence_exact(input_points, outputs):
    # requires three backward passes instead one like divergence_approx
    jac = _get_minibatch_jacobian(outputs, input_points)
    diagonal = jac.view(jac.shape[0], -1)[:, :: (jac.shape[1]+1)]
    return torch.sum(diagonal, 1)


def PDE_EQs(D_t, D_x, D_y, D_z, U, U_t=None, U_x=None, U_y=None, U_z=None):
    eqs = []
    dts = [D_t] 
    dxs = [D_x] 
    dys = [D_y] 
    dzs = [D_z] 

    if None not in [U_t, U_x, U_y, U_z]:
        dts += U_t.split(1, dim = -1) # [d_t, u_t, v_t, w_t] # (N,1)
        dxs += U_x.split(1, dim = -1) # [d_x, u_x, v_x, w_x]
        dys += U_y.split(1, dim = -1) # [d_y, u_y, v_y, w_y]
        dzs += U_z.split(1, dim = -1) # [d_z, u_z, v_z, w_z]
        
    u,v,w = U.split(1, dim=-1) # (N,1)
    for dt, dx, dy, dz in zip (dts, dxs, dys, dzs):
        _e = dt + (u*dx + v*dy + w*dz)
        eqs += [_e]
    # transport and nse equations:
    # e1 = d_t + (u*d_x + v*d_y + w*d_z) - PecInv*(c_xx + c_yy + c_zz)          , should = 0
    # e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - ReyInv*(u_xx + u_yy + u_zz)    , should = 0
    # e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - ReyInv*(v_xx + v_yy + v_zz)    , should = 0
    # e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - ReyInv*(w_xx + w_yy + w_zz)    , should = 0
    # e5 = u_x + v_y + w_z                                                      , should = 0
    # For simplification, we assume PecInv = 0.0, ReyInv = 0.0, pressure p = (0,0,0)                      
    
    if None not in [U_t, U_x, U_y, U_z]:
        # eqs += [ u_x + v_y + w_z ]
        eqs += [ dxs[1] + dys[2] + dzs[3] ]

    if True: # scale regularization
        eqs += [ (u*u + v*v + w*w)* 1e-1]
    
    return eqs

#####################################################################
# Coord Tools (all for torch Tensors)
# Coords:
# 1. resolution space, Frames x Depth x H x W, coord (frame_t, voxel_z, voxel_y, voxel_x),
# 2. simulation space, scale the resolution space to around 0-1, 
#    (FrameLength and Width in [0-1], Height and Depth keep ratios wrt Width)
# 3. target space, 
# 4. world space,
# 5. camera spaces,

# Vworld, Pworld; velocity, position in 4. world coord.
# Vsmoke, Psmoke; velocity, position in 2. simulation coord.
# w2s, 4.world to 3.target matrix (vel transfer uses rotation only; pos transfer includes offsets)
# s2w, 3.target to 4.world matrix (vel transfer uses rotation only; pos transfer includes offsets)
# scale_vector, to scale from 2.simulation space to 3.target space (no rotation, no offset)
#        for synthetic data, scale_vector = openvdb voxel size * [W,H,D] grid resolution (x first, z last), 
#        for e.g., scale_vector = 0.0469 * 256 = 12.0064
# st_factor, spatial temporal resolution ratio, to scale velocity from 2.simulation unit to 1.resolution unit
#        for e.g.,  st_factor = [W/float(max_timestep),H/float(max_timestep),D/float(max_timestep)]

# functions to transfer between 4. world space and 2. simulation space, 
# velocity are further scaled according to resolution as in mantaflow
def vel_world2smoke(Vworld, w2s, st_factor):
    vel_rot = Vworld[..., None, :] * (w2s[:3, :3])
    vel_rot = torch.sum(vel_rot, -1)  # 4.world to 3.target
    vel_scale = vel_rot * st_factor  # 3.target to 2.simulation
    return vel_scale


def vel_smoke2world(Vsmoke, s2w, st_factor):
    vel_scale = Vsmoke / st_factor  # 2.simulation to 3.target
    vel_rot = torch.sum(vel_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    return vel_rot


def pos_world2smoke(Pworld, w2s):
    # pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3,:3]), -1) # 4.world to 3.target
    pos_rot = (w2s[:3, :3] @ Pworld[..., :, None]).squeeze()
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
    new_pose = pos_rot + pos_off
    return new_pose


def off_smoke2world(Offsmoke, s2w):
    off_scale = Offsmoke  # 2.simulation to 3.target
    off_rot = torch.sum(off_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    return off_rot


def pos_smoke2world(Psmoke, s2w):
    pos_scale = Psmoke  # 2.simulation to 3.target
    pos_rot = torch.sum(pos_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    pos_off = (s2w[:3, -1]).expand(pos_rot.shape)  # 3.target to 4.world
    return pos_rot + pos_off


def get_voxel_pts(H, W, D, s2w, n_jitter=0, r_jitter=0.8):
    """Get voxel positions."""

    i, j, k = torch.meshgrid(
        torch.linspace(0, D - 1, D),
        torch.linspace(0, H - 1, H),
        torch.linspace(0, W - 1, W))
    pts = torch.stack([(k + 0.5) / W, (j + 0.5) / H, (i + 0.5) / D], -1).to(s2w.device)
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_jitter / W, r_jitter / H, r_jitter / D]).float().expand(pts.shape).to(s2w.device)
    for i_jitter in range(n_jitter):
        off_i = torch.rand(pts.shape, dtype=torch.float) - 0.5
        # shape D*H*W*3, value [(x,y,z)] , range [-0.5,0.5]

        pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w)


def get_density_flat(cur_pts, chunk=1024*32, network_fn: RadianceField = None, getStatic=True, **kwargs):
    input_shape = list(cur_pts.shape[0:-1])

    pts_flat = cur_pts.view(-1, cur_pts.shape[-1])
    pts_N = pts_flat.shape[0]
    # Evaluate model
    all_sigma = []
    for i in range(0, pts_N, chunk):
        pts_i = pts_flat[i:i+chunk]

        if isinstance(network_fn, HybridRadianceField):
            # kwargs["opaque"] = True
            raw_i = network_fn.static_model.query_density(pts_i[..., :3], **kwargs)
            raw_j = network_fn.dynamic_model.query_density(pts_i)
            all_sigma.append(torch.cat([raw_i, raw_j], -1))
        else:
            raw_i = network_fn.query_density(pts_i)
            all_sigma.append(raw_i)

    all_sigma = torch.cat(all_sigma, 0).view(input_shape+[-1])
    den_raw = all_sigma[..., -1:]
    returnStatic = getStatic and (all_sigma.shape[-1] > 1)
    if returnStatic:
        static_raw = all_sigma[..., :1]
        return [den_raw, static_raw]
    return [den_raw]


def get_velocity_flat(cur_pts, chunk=1024*32, vel_model=None):
    pts_N = cur_pts.shape[0]
    world_v = []
    for i in range(0, pts_N, chunk):
        input_i = cur_pts[i:i+chunk]
        vel_i = vel_model(input_i)
        world_v.append(vel_i)
    world_v = torch.cat(world_v, 0)
    return world_v


def get_density_and_derivatives(cur_pts, chunk=1024*32, network_fn=None, **kwargs):
    _den = get_density_flat(cur_pts, chunk, network_fn, False, **kwargs)[0]
    # requires 1 backward passes
    # The minibatch Jacobian matrix of shape (N, D_y=1, D_x=4)
    jac = _get_minibatch_jacobian(_den, cur_pts)
    jac = torch.where(torch.isnan(jac), 0.0, jac)   # fix for s-density in neus
    # assert not torch.any(torch.isnan(jac))
    _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,1)
    return _den, _d_x, _d_y, _d_z, _d_t


def get_velocity_and_derivatives(cur_pts, chunk=1024*32, vel_model=None):
    _vel = get_velocity_flat(cur_pts, chunk, vel_model)
    # requires 3 backward passes
    # The minibatch Jacobian matrix of shape (N, D_y=3, D_x=4)
    jac = _get_minibatch_jacobian(_vel, cur_pts)
    _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
    return _vel, _u_x, _u_y, _u_z, _u_t
    

class VoxelTool(object):

    def __get_tri_slice(self, _xm, _ym, _zm, _n=1):
        _yz = torch.reshape(self.pts[...,_xm:_xm+_n,:],(-1,3))
        _zx = torch.reshape(self.pts[:,_ym:_ym+_n,...],(-1,3))
        _xy = torch.reshape(self.pts[_zm:_zm+_n,...],(-1,3))
        
        pts_mid = torch.cat([_yz, _zx, _xy], dim=0)
        npMaskXYZ = [np.zeros([self.D,self.H,self.W,1], dtype=np.float32) for _ in range(3)]
        npMaskXYZ[0][...,_xm:_xm+_n,:] = 1.0
        npMaskXYZ[1][:,_ym:_ym+_n,...] = 1.0
        npMaskXYZ[2][_zm:_zm+_n,...] = 1.0
        return pts_mid, torch.tensor(np.clip(npMaskXYZ[0]+npMaskXYZ[1]+npMaskXYZ[2], 1e-6, 3.0), device=pts_mid.device)

    def __pad_slice_to_volume(self, _slice, _n, mode=0):
        # mode: 0, x_slice, 1, y_slice, 2, z_slice
        tar_shape = [self.D,self.H,self.W]
        in_shape = tar_shape[:]
        in_shape[-1-mode] = _n
        fron_shape = tar_shape[:]
        fron_shape[-1-mode] = (tar_shape[-1-mode] - _n)//2
        back_shape = tar_shape[:]
        back_shape[-1-mode] = (tar_shape[-1-mode] - _n - fron_shape[-1-mode])

        cur_slice = _slice.view(in_shape+[-1])
        front_0 = torch.zeros(fron_shape + [cur_slice.shape[-1]], device=_slice.device)
        back_0 = torch.zeros(back_shape + [cur_slice.shape[-1]], device=_slice.device)

        volume = torch.cat([front_0, cur_slice, back_0], dim=-2-mode)
        return volume

    def __init__(self, voxel_tran: torch.Tensor, voxel_tran_inv: torch.Tensor, scene_scale: np.ndarray,
                 x: int, middle_view: bool = True):
        assert scene_scale[0] == 1.0    # normalized by x /= x[0]
        scene_size = (scene_scale * x).round().astype(int)
        W, H, D = scene_size
        self.s_s2w = voxel_tran
        self.s_w2s = voxel_tran_inv
        self.D = D
        self.H = H
        self.W = W
        self.pts = get_voxel_pts(H, W, D, self.s_s2w)
        self.pts_mid = None
        self.mask_xyz = None
        self.middle_view = middle_view
        if middle_view is not None:
            _n = 1 if middle_view else 3
            _xm, _ym, _zm = (W - _n) // 2, (H - _n) // 2, (D - _n) // 2
            self.pts_mid, self.mask_xyz = self.__get_tri_slice(_xm, _ym, _zm, _n)

    def voxel_size(self) -> tuple[int, int, int]:
        return self.W, self.H, self.D
        
    def get_voxel_density_list(self,t=None,chunk=1024*32, network_fn=None, middle_slice=False, **kwargs):
        D,H,W = self.D,self.H,self.W
        # middle_slice, only for fast visualization of the middle slice
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        if t is not None:
            pts_flat = attach_time(pts_flat, t)

        den_list = get_density_flat(pts_flat, chunk, network_fn, **kwargs)

        return_list = []
        for den_raw in den_list:
            if middle_slice:
                # only for fast visualization of the middle slice
                _n = 1 if self.middle_view else 3
                _yzV, _zxV, _xyV = torch.split(den_raw, [D*H*_n,D*W*_n,H*W*_n], dim=0)
                mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) +self.__pad_slice_to_volume(_xyV, _n, 2)
                return_list.append(mixV / self.mask_xyz)
            else:
                return_list.append(den_raw.view(D,H,W,1))
        return return_list
        
    def get_voxel_velocity(self,deltaT,t,chunk=1024*32,
        vel_model=None, middle_slice=False, ref_den_list=None):
        # middle_slice, only for fast visualization of the middle slice
        D,H,W = self.D,self.H,self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        if t is not None:
            pts_flat = attach_time(pts_flat, t)

        world_v = get_velocity_flat(pts_flat,chunk,vel_model)
        reso_scale = torch.tensor([self.W*deltaT,self.H*deltaT,self.D*deltaT], device=pts_flat.device)
        target_v = vel_world2smoke(world_v, self.s_w2s, reso_scale)

        if middle_slice:
            _n = 1 if self.middle_view else 3
            _yzV, _zxV, _xyV = torch.split(target_v, [D*H*_n,D*W*_n,H*W*_n], dim=0)
            mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) +self.__pad_slice_to_volume(_xyV, _n, 2)
            target_v = mixV / self.mask_xyz
        else:
            target_v = target_v.view(D,H,W,3)

        if ref_den_list is not None:
            target_v = target_v - target_v * torch.less(ref_den_list, 0.1) * 0.5
        
        return target_v

    def save_voxel_den_npz(self,den_path,t,network_fn=None,chunk=1024*32,save_npz=True,save_jpg=False, jpg_mix=True,
                           noStatic=False, **kwargs):
        voxel_den_list = self.get_voxel_density_list(t,chunk, network_fn, middle_slice=not (jpg_mix or save_npz), **kwargs)
        head_tail = os.path.split(den_path)
        namepre = ["","static_"]
        for voxel_den, npre in zip(voxel_den_list, namepre):
            voxel_den = voxel_den.detach().cpu().numpy()
            if save_jpg:
                jpg_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+".jpg")
                imageio.imwrite(jpg_path, den_scalar2rgb(voxel_den, scale=None, is3D=True, logv=False, mix=jpg_mix).squeeze())
            if save_npz:
                # to save some space
                npz_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0]+".npz")
                voxel_den = np.float16(voxel_den)
                np.savez_compressed(npz_path, vel=voxel_den)
            if noStatic:
                break

    def save_voxel_vel_npz(self,vel_path,deltaT,t,chunk=1024*32, vel_model=None,save_npz=True,save_jpg=False,save_vort=False):
        vel_scale = 160
        voxel_vel = self.get_voxel_velocity(deltaT,t,chunk,vel_model,middle_slice=not save_npz).detach().cpu().numpy()
        
        if save_jpg:
            jpg_path = os.path.splitext(vel_path)[0]+".jpg"
            imageio.imwrite(jpg_path, vel_uv2hsv(voxel_vel, scale=vel_scale, is3D=True, logv=False))
        if save_npz:
            if save_vort and save_jpg:
                _, NETw = jacobian3D_np(voxel_vel)
                head_tail = os.path.split(vel_path)
                imageio.imwrite( os.path.join(head_tail[0], "vort"+os.path.splitext(head_tail[1])[0]+".jpg"),
                        vel_uv2hsv(NETw[0],scale=vel_scale*5.0,is3D=True) )
            # to save some space
            voxel_vel = np.float16(voxel_vel)
            np.savez_compressed(vel_path, vel=voxel_vel)


def voxel_output(writer: VoxelTool, model: RadianceField, prop_model: RadianceField | None, vel_model, img_dir: str,
                 step: int, timestep: float = None, scale: float = None):
    voxel_den_list = writer.get_voxel_density_list(t=timestep, network_fn=model, middle_slice=False, opaque=True)[::-1]
    smoke_den = voxel_den_list[-1]
    if prop_model is not None:
        voxel_den_list += writer.get_voxel_density_list(t=timestep, network_fn=prop_model, middle_slice=False)
    if vel_model is not None:
        voxel_den_list.append(
            writer.get_voxel_velocity(0.01, 0.5, vel_model=vel_model, middle_slice=True, ref_den_list=smoke_den))
    voxel_img = []
    for voxel in voxel_den_list:
        voxel = voxel.detach().cpu().numpy()
        if voxel.shape[-1] == 1:
            voxel_img.append(
                np.repeat(den_scalar2rgb(voxel, scale=scale, is3D=True, logv=False, mix=True), 3, axis=-1))
        else:
            voxel_img.append(vel_uv2hsv(voxel, scale=300, is3D=True, logv=False))
    voxel_img = np.concatenate(voxel_img, axis=0)  # 128,64*3,3
    imageio.imwrite(os.path.join(img_dir, 'vox_{:06d}.png'.format(step)), voxel_img)


#####################################################################
# Loss Tools (all for torch Tensors)
def fade_in_weight(step, start, duration):
    return min(max((float(step) - start)/duration, 0.0), 1.0)


# Ghost Density Loss Tool
def ghost_loss_func(out: NeRFOutputs, bg: torch.Tensor, scale: float = 4.0):
    ghost_mask = torch.mean(torch.square(out.rgb - bg), -1)
    # ghost_mask = torch.sigmoid(ghost_mask*-1.0) + den_penalty # (0 to 0.5) + den_penalty
    ghost_mask = torch.exp(ghost_mask * -scale)
    ghost_alpha = ghost_mask * out.acc
    return torch.mean(torch.square(ghost_alpha))


# original version https://docs.opencv.org/4.x/d5/dc4/tutorial_video_input_psnr_ssim.html
def get_MSSIM(I1: np.ndarray, I2: np.ndarray):
    C1 = 6.5025
    C2 = 58.5225
    # INITS

    I1 = np.float32(I1)  # cannot calculate on one byte large values
    I2 = np.float32(I2)

    I2_2 = I2 * I2  # I2^2
    I1_2 = I1 * I1  # I1^2
    I1_I2 = I1 * I2  # I1 * I2
    # END INITS

    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2

    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2

    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    ssim_map = cv.divide(t3, t1)  # ssim_map = t3./t1;

    mssim = cv.mean(ssim_map)  # mssim = average of ssim map
    return mssim[:3]