import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn

from .neus import SingleVarianceNetwork, SDFRadianceField


class NSR(SDFRadianceField):
    def __init__(self,
                 input_ch=3,
                 input_ch_views=3,
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 geometric_init=True,
                 weight_norm=True,
                 include_input=True,
                 bound=1.0,
                 init_variance=0.3,
                 output_s_density=True,
                 solid_density=5.0,
                 fading_fin_step=1,
                 **kwargs,
                 ):
        super().__init__()

        # sdf network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.include_input = include_input
        self.bound = bound
        self.output_s_density = output_s_density
        self.solid_density = solid_density

        self.encoder = tcnn.Encoding(
            n_input_dims=input_ch,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            }
        )
        self.in_dim = self.encoder.n_output_dims

        sdf_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = (self.in_dim + input_ch) if self.include_input else self.in_dim

            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sdf_net.append(nn.Linear(in_dim, out_dim))

            if geometric_init:
                if l == num_layers - 1:
                    nn.init.normal_(sdf_net[l].weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(sdf_net[l].bias, 0)

                elif l == 0:
                    if self.include_input:
                        nn.init.constant_(sdf_net[l].bias, 0.0)
                        nn.init.normal_(sdf_net[l].weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                        nn.init.constant_(sdf_net[l].weight[:, 3:], 0.0)
                    else:
                        nn.init.constant_(sdf_net[l].bias, 0.0)
                        nn.init.normal_(sdf_net[l].weight[:, :], 0.0, np.sqrt(2) / np.sqrt(out_dim))

                else:
                    nn.init.constant_(sdf_net[l].bias, 0.0)
                    nn.init.normal_(sdf_net[l].weight[:, :], 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                sdf_net[l] = nn.utils.weight_norm(sdf_net[l])

        self.sdf_net = nn.ModuleList(sdf_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.use_viewdirs = input_ch_views > 0
        if self.use_viewdirs:
            self.encoder_dir = tcnn.Encoding(
                n_input_dims=input_ch_views,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                }
            )
            self.in_dim_color = self.encoder_dir.n_output_dims
        else:
            self.encoder_dir = None
            self.in_dim_color = 0
        self.in_dim_color = self.in_dim_color + self.geo_feat_dim + 6  # hash_feat + dir + geo_feat + normal(sdf gradiant) 32 +

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

            if weight_norm:
                color_net[l] = nn.utils.weight_norm(color_net[l])

        self.color_net = nn.ModuleList(color_net)
        self.deviation_net = SingleVarianceNetwork(init_variance)

        self.activation = nn.Softplus(beta=100)
        self.fading_step = 1
        self.fading_fin_step = max(1, fading_fin_step)

    def update_fading_step(self, fading_step):
        if fading_step > 1:
            self.fading_step = fading_step

    def progress_lambda(self):
        lam = int(self.fading_step / self.fading_fin_step * 16)
        return min(16, lam)

    def print_fading(self):
        w_list = self.progress_lambda()
        print(f"lambda = {w_list}")

    def forward_sdf(self, x):
        # x: [B, N, 3], in [-bound, bound]
        # sdf
        # TODO handle bound
        # x = (x + self.bound) / (2 * self.bound)
        x = x / self.bound
        h = self.encoder(torch.reshape(x, (-1, 3)))     # encoder only accept [N, 3]
        h = h.reshape(*x.shape[:-1], -1)

        lam = self.progress_lambda()
        h, h_zero = torch.split(h, [lam * 2, h.shape[-1] - lam * 2], dim=-1)
        h_zero = torch.zeros_like(h_zero)

        if self.include_input:
            h = torch.cat([x, h, h_zero], dim=-1)
        else:
            h = torch.cat([h, h_zero], dim=-1)

        for l in range(self.num_layers):
            h = self.sdf_net[l](h)
            if l != self.num_layers - 1:
                h = self.activation(h)
                # h = F.relu(h, inplace=True)
        sdf_output = h

        return sdf_output

    def forward_color(self, x, d, n, geo_feat):

        if self.use_viewdirs:
            # dir
            d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
            d = self.encoder_dir(d.reshape(-1, 3)).reshape(*d.shape[:-1], -1)
            # color x,
            h = torch.cat([x, d, n, geo_feat], dim=-1)
        else:
            h = torch.cat([x, n, geo_feat], dim=-1)

        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return color

    def deviation(self) -> torch.Tensor:
        return self.deviation_net.forward()

    def sdf(self, x):
        return self.forward_sdf(x)[..., :1]

    def gradient(self, x, epsilon=0.0005, **kwargs):
        # not allowed auto gradient, using fd instead
        return self.finite_difference_normals_approximator(x, epsilon)

    def finite_difference_normals_approximator(self, x, epsilon=0.0005):
        # finite difference
        # f(x+h, y, z), f(x, y+h, z), f(x, y, z+h) - f(x-h, y, z), f(x, y-h, z), f(x, y, z-h)
        pos_x = x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_pos = self.sdf(pos_x)
        pos_y = x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)
        dist_dy_pos = self.sdf(pos_y)
        pos_z = x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)
        dist_dz_pos = self.sdf(pos_z)

        neg_x = x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)
        dist_dx_neg = self.sdf(neg_x)
        neg_y = x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)
        dist_dy_neg = self.sdf(neg_y)
        neg_z = x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)
        dist_dz_neg = self.sdf(neg_z)

        return torch.cat([dist_dx_pos - dist_dx_neg, dist_dy_pos - dist_dy_neg, dist_dz_pos - dist_dz_neg],
                         dim=-1) * (0.5 / epsilon)

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

    def forward(self, input_pts, input_views, cond=None):
        sdf_nn_output = self.forward_sdf(input_pts)
        sdf = sdf_nn_output[..., :1]
        feature_vector = sdf_nn_output[..., 1:]

        gradients = self.gradient(input_pts)
        normal = gradients / (1e-5 + torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True))

        color = self.forward_color(input_pts, input_views, normal, feature_vector)

        inv_s = self.deviation()     # Single parameter

        if self.output_s_density:
            sigma = self.s_density(sdf, inv_s)
        else:
            sigma = None
        return color, sigma, {
            'sdf': sdf,
            'gradients': gradients,
            'inv_s': inv_s
        }
