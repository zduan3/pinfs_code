import torch
import torch.nn as nn
import torchvision
import numpy as np


# VGG Tool, https://github.com/crowsonkb/style-transfer-pytorch/
# set pooling = 'max'
class VGGFeatures(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = torchvision.models.vgg19(pretrained=True).features[:self.layers[-1] + 1]

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                             stride=conv.stride, padding=conv.padding,
                             padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def forward(self, input, layers=None):
        # input shape, b,3,h,w
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        norm_in = torch.stack([self.normalize(input[_i]) for _i in range(input.shape[0])], dim=0)
        # input = self.normalize(input)
        for i in range(max(layers) + 1):
            norm_in = self.model[i](norm_in)
            if i in layers:
                feats[i] = norm_in
        return feats


# VGG Loss Tool
class VGGLossTool(object):
    def __init__(self, device):
        # The default content and style layers in Gatys et al. (2015):
        #   content_layers = [22], 'relu4_2'
        #   style_layers = [1, 6, 11, 20, 29], relu layers: [ 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        # We use [5, 10, 19, 28], conv layers before relu: [ 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.layer_list = [5, 10, 19, 28]
        self.layer_names = [
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.device = device

        # Build a VGG19 model loaded with pre-trained ImageNet weights
        self.vggmodel = VGGFeatures(self.layer_list).to(device)

    def feature_norm(self, feature):
        # feature: b,h,w,c
        feature_len = torch.sqrt(torch.sum(torch.square(feature), dim=-1, keepdim=True) + 1e-12)
        norm = feature / feature_len
        return norm

    def cos_sim(self, a, b):
        cos_sim_ab = torch.sum(a * b, dim=-1)
        # cosine similarity, -1~1, 1 best
        cos_sim_ab_score = 1.0 - torch.mean(cos_sim_ab)  # 0 ~ 2, 0 best
        return cos_sim_ab_score

    def compute_cos_loss(self, img, ref):
        # input img, ref should be in range of [0,1]
        input_tensor = torch.stack([ref, img], dim=0)

        input_tensor = input_tensor.permute((0, 3, 1, 2))
        # print(input_tensor.shape)
        _feats = self.vggmodel(input_tensor, layers=self.layer_list)

        # Initialize the loss
        loss = []
        # Add loss
        for layer_i, layer_name in zip(self.layer_list, self.layer_names):
            cur_feature = _feats[layer_i]
            reference_features = self.feature_norm(cur_feature[0, ...])
            img_features = self.feature_norm(cur_feature[1, ...])

            feature_metric = self.cos_sim(reference_features, img_features)
            loss += [feature_metric]
        return loss


def vgg_sample(vgg_strides: int, num_rays: int, frame: torch.Tensor, bg_color: torch.Tensor, dw: int = None,
               steps: int = None):
    if steps is None:
        strides = vgg_strides + np.random.randint(-1, 2)  # args.vgg_strides(+/-)1 or args.vgg_strides
    else:
        strides = vgg_strides + steps % 3 - 1
    H, W = frame.shape[:2]
    if dw is None:
        dw = max(20, min(40, int(np.sqrt(num_rays))))
    vgg_min_border = 10
    strides = min(strides, min(H - vgg_min_border, W - vgg_min_border) / dw)
    strides = int(strides)

    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing='ij'),
                         dim=-1).to(frame.device)  # (H, W, 2)
    target_grey = torch.mean(torch.abs(frame - bg_color), dim=-1, keepdim=True)  # (H, W, 1)
    img_wei = coords.to(torch.float32) * target_grey
    center_coord = torch.sum(img_wei, dim=(0, 1)) / torch.sum(target_grey)
    center_coord = center_coord.cpu().numpy()
    # add random jitter
    random_R = dw * strides / 2.0
    # mean and standard deviation: center_coord, random_R/3.0, so that 3sigma < random_R
    random_x = np.random.normal(center_coord[1], random_R / 3.0) - 0.5 * dw * strides
    random_y = np.random.normal(center_coord[0], random_R / 3.0) - 0.5 * dw * strides

    offset_w = int(min(max(vgg_min_border, random_x), W - dw * strides - vgg_min_border))
    offset_h = int(min(max(vgg_min_border, random_y), H - dw * strides - vgg_min_border))

    coords_crop = coords[offset_h:offset_h + dw * strides:strides, offset_w:offset_w + dw * strides:strides, :]
    return coords_crop, dw
