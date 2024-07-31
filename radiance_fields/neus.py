import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .nerf import positional_encoder
from .radiance_field import RadianceField


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self):
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


class SDFRadianceField(RadianceField):
    def __init__(self):
        super().__init__(True)

    @abc.abstractmethod
    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """ output sdf as [shape, 1] """
        pass

    @abc.abstractmethod
    def gradient(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """ output sdf gradient as [shape, 3] """
        pass

    # NeuS specific

    @abc.abstractmethod
    def deviation(self) -> torch.Tensor:
        """get inv_s as standard deviation"""
        pass

    def s_density(self, sdf: torch.Tensor, inv_s: torch.Tensor = None):
        if inv_s is None:
            inv_s = self.deviation()
        exp_sx = torch.exp(-sdf * inv_s)
        return inv_s * exp_sx / (1 + exp_sx) ** 2

    def opaque_density(self, sdf: torch.Tensor, inv_s: torch.Tensor = None):
        if inv_s is None:
            inv_s = self.deviation()
        rho = inv_s / (torch.exp(inv_s * sdf) + 1)  # phi_s(x) / Phi_s(x)
        return torch.clip(rho, max=self.solid_density)

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs):
        if kwargs.get('opaque', False):
            return self.opaque_density(self.sdf(x))
        return self.s_density(self.sdf(x))


class NeuS(SDFRadianceField):
    def __init__(self, D=8, W=256, input_ch=3, use_viewdirs=True, skips=(4,), n_features=256,
                 multires=0, multires_views=0, geometric_init=True, init_bias=0.5, bound=1.0, use_color_t=False,
                 output_s_density=True, init_variance=0.3, solid_density=5.0, fading_fin_step=1):
        super().__init__()

        self.input_ch = input_ch
        self.use_viewdirs = use_viewdirs > 0

        self.embed_fn, self.input_ch = positional_encoder(multires, input_dims=input_ch)
        self.embed_fn_views, self.input_ch_views = positional_encoder(multires_views)

        dims = [self.input_ch] + [W for _ in range(D-1)] + [1 + n_features]
        self.num_layers = D
        self.skips = skips

        sdf_net = []
        for l in range(0, self.num_layers):
            if l + 1 in self.skips:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 1:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi / dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -init_bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2 / out_dim))
                elif multires > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2 / out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2 / out_dim))
            lin = nn.utils.weight_norm(lin)

            sdf_net.append(lin)

        self.sdf_net = nn.ModuleList(sdf_net)
        self.activation = nn.Softplus(beta=100)

        self.use_color_t = use_color_t
        self.embed_t = None
        color_in_dims = 9 if self.use_viewdirs else 6
        if use_color_t:
            # self.embed_t, t_dim = positional_encoder(multires, input_dims=1)
            self.embed_t, t_dim = positional_encoder(0, input_dims=1)
            color_in_dims += t_dim
        dims = [n_features + color_in_dims] + [W for _ in range(4)] + [3]
        self.num_layers_color = 5

        color_net = []
        for l in range(0, self.num_layers_color):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            lin = nn.utils.weight_norm(lin)

            color_net.append(lin)

        self.color_net = nn.ModuleList(color_net)
        self.deviation_network = SingleVarianceNetwork(init_variance)

        self.bound = bound
        self.output_s_density = output_s_density
        self.solid_density = solid_density
        self.fading_step = 1
        self.fading_fin_step = max(1, fading_fin_step)
        self.multires = multires

    def deviation(self) -> torch.Tensor:
        return self.deviation_network()

    def update_fading_step(self, fading_step):
        if fading_step > 1:
            self.fading_step = fading_step

    def fading_wei_list(self):
        wei_list = [1.0]
        alpha = self.fading_step * self.multires / self.fading_fin_step
        for freq_n in range(self.multires):
            w_a = (1.0 - np.cos(np.pi * np.clip(alpha - freq_n, 0, 1))) * 0.5
            wei_list += [w_a, w_a]  # sin, cos
        return wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%.3f" % (i * 3, w_list[i])
                for i in range(len(w_list)) if 1e-5 < w_list[i] < 1 - 1e-5]
        print("; ".join(_str))

    def forward_sdf(self, inputs):
        inputs = inputs / self.bound    # map to [0, 1]
        # inputs = inputs * 2 / self.bound - 1.0  # map to [-1, 1]
        inputs = self.embed_fn(inputs)

        if self.multires > 0 and self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            pts = torch.zeros_like(inputs)
            for i, wei in enumerate(fading_wei_list):
                if wei > 1e-8:
                    bgn = i * 3
                    end = bgn + 3  # full fading
                    pts[..., bgn: end] = inputs[..., bgn: end] * wei
            inputs = pts

        x = inputs
        for l in range(0, self.num_layers):
            lin = self.sdf_net[l]

            if l in self.skips:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 1:
                x = self.activation(x)
        return x

    def sdf(self, x):
        return self.forward_sdf(x)[..., :1]

    def forward_with_gradient(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            output = self.forward_sdf(x)
            y = output[..., :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return output, gradients

    def gradient(self, x, **kwargs):
        return self.forward_with_gradient(x)[1]

    def forward_color(self, points, normals, view_dirs, color_t, feature_vectors):
        cat_list = [points]
        if self.use_viewdirs:
            view_dirs = self.embed_fn_views(view_dirs)
            cat_list.append(view_dirs)
        if self.use_color_t:
            cat_list.append(self.embed_t(color_t))
        cat_list += [normals, feature_vectors]
        rendering_input = torch.cat(cat_list, dim=-1)

        x = rendering_input
        for l in range(0, self.num_layers_color):
            lin = self.color_net[l]

            x = lin(x)
            if l < self.num_layers_color - 1:
                x = F.relu(x)

        x = torch.sigmoid(x)
        return x

    def forward(self, input_pts, input_views, cond: torch.Tensor = None):
        sdf_nn_output, gradients = self.forward_with_gradient(input_pts)
        sdf = sdf_nn_output[..., :1]
        feature_vectors = sdf_nn_output[..., 1:]

        # x, n, v, z in IDR
        sampled_color = self.forward_color(input_pts, gradients, input_views, cond, feature_vectors)
        inv_s = self.deviation_network()

        if self.output_s_density:
            sigma = self.s_density(sdf, inv_s)
            # sigma = self.opaque_density(sdf, inv_s)
        else:
            sigma = None
        return sampled_color, sigma, {
            'sdf': sdf,
            'gradients': gradients,
            'inv_s': inv_s
        }
