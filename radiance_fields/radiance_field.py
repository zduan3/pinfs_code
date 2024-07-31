import abc
import torch
import torch.nn as nn


class RadianceField(nn.Module):
    def __init__(self, output_sdf: bool = False):
        """
        Args:
            output_sdf: indicate that the returned extra part of forward() contains sdf
        """
        super().__init__()
        self.output_sdf = output_sdf

    @abc.abstractmethod
    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, dirs: torch.Tensor | None, cond: torch.Tensor = None) \
            -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """
        Args:
            x: input points [shape, 3 or 4]
            dirs: input directions [shape, 3]
            cond: extra conditions
        Returns:
            rgb [shape, 3], sigma [shape, 1] if applicable, extra outputs as dict
        """
        pass

    # pinf fading support, optional

    def update_fading_step(self, fading_step: int):
        pass

    def print_fading(self):
        pass


class HybridRadianceField(RadianceField):
    def __init__(self, static_model: RadianceField, dynamic_model: RadianceField):
        super().__init__(static_model.output_sdf)
        self.static_model = static_model
        self.dynamic_model = dynamic_model

    def update_fading_step(self, fading_step: int):
        self.static_model.update_fading_step(fading_step)
        self.dynamic_model.update_fading_step(fading_step)

    def print_fading(self):
        print('static: ', end='')
        self.static_model.print_fading()
        print('dynamic: ', end='')
        self.dynamic_model.print_fading()

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        s_static = self.static_model.query_density(x[..., :3], cond, **kwargs)
        s_dynamic = self.dynamic_model.query_density(x, cond)
        return s_static + s_dynamic

    def forward(self, x: torch.Tensor, dirs: torch.Tensor | None, cond: torch.Tensor = None):
        rgb_s, sigma_s, extra_s = self.static_model.forward(x[..., :3], dirs, cond)
        rgb_d, sigma_d, extra_d = self.dynamic_model.forward(x, dirs, cond)
        return self.merge_result(self.output_sdf, rgb_s, sigma_s, extra_s, rgb_d, sigma_d, extra_d)

    @staticmethod
    def merge_result(output_sdf: bool, rgb_s, sigma_s, extra_s, rgb_d, sigma_d, extra_d):
        if output_sdf:
            sigma = sigma_d
            rgb = rgb_d
        else:   # does alpha blend, when delta -> 0
            sigma = sigma_s + sigma_d
            rgb = (rgb_s * sigma_s + rgb_d * sigma_d) / (sigma + 1e-6)

        extra_s |= {
            'rgb_s': rgb_s,
            'rgb_d': rgb_d,
            'sigma_s': sigma_s,
            'sigma_d': sigma_d,
        }
        if len(extra_d) > 0:
            extra_s['dynamic'] = extra_d
        return rgb, sigma, extra_s
