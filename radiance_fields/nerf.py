import torch
from torch import nn
import torch.nn.functional as F

from .radiance_field import RadianceField


class InputEncoder(nn.Module):
    def __init__(self, input_dims: int):
        super().__init__()
        self.output_dims = input_dims

    def forward(self, inputs: torch.Tensor):
        return inputs


class SinusoidalEncoder(InputEncoder):
    def __init__(self, input_dims: int, num_freq: int, include_input: bool = True):
        super().__init__(input_dims)
        self.include_input = include_input
        self.freq_bands = 2 ** torch.linspace(0, num_freq - 1, steps=num_freq)
        self.output_dims = 2 * num_freq * input_dims
        if include_input:
            self.output_dims += input_dims

    def forward(self, inputs: torch.Tensor):
        outputs = []
        if self.include_input:
            outputs.append(inputs)
        for freq in self.freq_bands:
            mult = inputs * freq
            outputs.append(torch.sin(mult))
            outputs.append(torch.cos(mult))
        return torch.cat(outputs, -1)


# Positional encoding (section 5.1)
def positional_encoder(multires: int, input_dims: int = 3) -> tuple[InputEncoder, int]:
    encoder = SinusoidalEncoder(input_dims, multires) if multires > 0 and input_dims > 0 else InputEncoder(input_dims)
    return encoder, encoder.output_dims


class NeRF(RadianceField):
    def __init__(self, D=8, W=256, input_ch=3, skips=(4,), use_viewdirs=False,
                 multires=10, multires_views=4):
        """
        """
        super(NeRF, self).__init__()
        self.input_ch = input_ch
        self.input_ch_views = 3 if use_viewdirs else 0
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.encoder, input_ch = positional_encoder(multires, input_ch)
        self.encoder_views, input_ch_views = positional_encoder(multires_views)
        # self.activation = nn.Softplus(beta=100)
        self.activation = nn.ReLU()

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)])
        # for linear in self.pts_linears:
        #     nn.init.xavier_uniform_(linear.weight)
        #     nn.init.zeros_(linear.bias)

        self.sigma_linear = nn.Linear(W, 1)
        # nn.init.xavier_uniform_(self.sigma_linear.weight)
        # nn.init.zeros_(self.sigma_linear.bias)

        if use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.rgb_linear = nn.Linear(W, 3)
            nn.init.xavier_uniform_(self.rgb_linear.weight)
            nn.init.zeros_(self.rgb_linear.bias)

    def query_density_and_feature(self, x: torch.Tensor, cond: torch.Tensor = None):
        input_pts = self.encoder(x)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.activation(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        sigma = self.sigma_linear(h)
        return self.activation(sigma), h

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.query_density_and_feature(x, cond)[0]

    def forward(self, x, dirs, cond: torch.Tensor = None):
        sigma, h = self.query_density_and_feature(x, cond)

        if self.use_viewdirs:
            input_views = self.encoder_views(dirs)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

        rgb = self.rgb_linear(h)

        return torch.sigmoid(rgb), sigma, {}
