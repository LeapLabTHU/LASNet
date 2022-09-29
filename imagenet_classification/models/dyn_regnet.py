import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation
from torchvision.models._utils import _make_divisible
import torch.nn.functional as F

__all__ = [
    "dyn_regnet_y_400mf",
    "dyn_regnet_y_800mf",
]


model_urls = {
    "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
    "regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
}


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def apply_mask(x, mask):
    b, c, h, w = x.shape
    _, g, hw_mask, _ = mask.shape
    if (g > 1) and (g != c):
        mask = mask.unsqueeze(1).repeat(1,c//g,1,1,1).transpose(1,2).reshape(b,c,hw_mask,hw_mask)
    return x * mask


class Masker(nn.Module):
    def __init__(self, in_channels, mask_channel_group, mask_size, feature_size=32, dilate_stride=1):
        super(Masker, self).__init__()
        self.mask_channel_group = mask_channel_group
        self.mask_size = mask_size
        self.conv2 = conv1x1(in_channels, mask_channel_group*2,bias=True)
        self.conv2_flops_pp = self.conv2.weight.shape[0] * self.conv2.weight.shape[1] + self.conv2.weight.shape[1]
        self.conv2.bias.data[:mask_channel_group] = 10.0
        self.conv2.bias.data[mask_channel_group+1:] = 1.0
        self.feature_size = feature_size
        self.expandmask = ExpandMask(stride=dilate_stride, mask_channel_group=mask_channel_group)

    def forward(self, x, temperature):
        mask =  F.adaptive_avg_pool2d(x, self.mask_size) if self.mask_size < x.shape[2] else x
        flops = mask.shape[1] * mask.shape[2] * mask.shape[3]
        
        mask = self.conv2(mask)
        flops += self.conv2_flops_pp * mask.shape[2] * mask.shape[3]
        
        b,c,h,w = mask.shape
        mask = mask.view(b,2,c//2,h,w)
        if self.training:
            mask = F.gumbel_softmax(mask, dim=1, tau=temperature, hard=True)
            mask = mask[:,0]
        else:
            mask = (mask[:,0]>=mask[:,1]).float()
        sparsity = mask.sum() / mask.numel()
        
        if h < self.feature_size:
            mask = F.interpolate(mask, size = (self.feature_size, self.feature_size))
        mask_dil = self.expandmask(mask)
        sparsity_dil = mask_dil.sum() / mask_dil.numel()
        
        return mask, mask_dil, sparsity, sparsity_dil, flops


class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1, mask_channel_group=1): 
        super(ExpandMask, self).__init__()
        self.stride=stride
        self.padding = padding
        self.mask_channel_group = mask_channel_group
        
    def forward(self, x):
        if self.stride > 1:
            self.pad_kernel = torch.zeros((self.mask_channel_group,1,self.stride, self.stride), device=x.device)
            self.pad_kernel[:,:,0,0] = 1
        self.dilate_kernel = torch.ones((self.mask_channel_group,self.mask_channel_group,1+2*self.padding,1+2*self.padding), device=x.device)

        x = x.float()
        
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5


class SimpleStemIN(ConvNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer
        )
        

class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = ConvNormActivation(
            width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer
        )
        layers["b"] = ConvNormActivation(
            w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = ConvNormActivation(
            w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        mask_channel_group=1,
        output_size=56,
        mask_spatial_granularity=1
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(
                width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)
        
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width
        self.conv1_flops_per_pixel = width_in*w_b
        self.conv2_flops_per_pixel = w_b*w_b*9 // g
        self.conv3_flops_per_pixel = w_b*width_out

        if self.proj is not None:
            self.downsample_flops = width_in * width_out

        self.output_size = output_size
        self.mask_spatial_granularity = mask_spatial_granularity
        self.mask_size = self.output_size // self.mask_spatial_granularity
        self.masker = Masker(width_in, mask_channel_group, self.mask_size, feature_size=output_size, dilate_stride=stride)

        self.stride=stride
        
    def forward(self, x: Tensor, temperature:float) -> Tensor:
        x, sparsity_list, sparsity_list_dil, flops_perc_list, flops = x
        identity = x
        
        mask, mask_dil, sparsity, sparsity_dil, mask_flops = self.masker(x, temperature)
        sparse_flops = mask_flops
        dense_flops = mask_flops

        if sparsity_list == None:
            sparsity_list = sparsity.unsqueeze(0)
        else:
            sparsity_list = torch.cat((sparsity_list, sparsity.unsqueeze(0)), dim=0)
        if sparsity_list_dil == None:
            sparsity_list_dil = sparsity_dil.unsqueeze(0)
        else:
            sparsity_list_dil = torch.cat((sparsity_list_dil, sparsity_dil.unsqueeze(0)), dim=0)
        
        if self.proj is not None:
            identity = self.proj(x)
            dense_flops = self.downsample_flops * identity.shape[2] * identity.shape[3]
            sparse_flops = self.downsample_flops * identity.shape[2] * identity.shape[3]
        else:
            identity = x
        
        x = self.f(x)
        
        b,c,h,w = x.shape
        h_conv1 = h * self.stride
        
        dense_flops += (self.conv1_flops_per_pixel * h_conv1 * h_conv1 + self.conv2_flops_per_pixel * h * w + self.conv3_flops_per_pixel * h * w)
        sparse_flops += (self.conv1_flops_per_pixel * h_conv1 * h_conv1 * sparsity_dil + self.conv2_flops_per_pixel * h * w * sparsity + self.conv3_flops_per_pixel * h * w * sparsity)
        
        x = apply_mask(x, mask)
        x = self.activation(x + identity)
        
        flops += sparse_flops
        perc = sparse_flops / dense_flops

        if flops_perc_list == None:
            flops_perc_list = perc.unsqueeze(0)
        else:
            flops_perc_list = torch.cat((flops_perc_list,perc.unsqueeze(0)),dim=0)
        return x, sparsity_list, sparsity_list_dil, flops_perc_list, flops


class AnyStage(nn.ModuleList):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
        mask_channel_group=[1],
        output_size=56,
        mask_spatial_granularity=[1]
    ) -> None:
        super().__init__()
        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
                mask_channel_group=mask_channel_group[i],
                output_size=output_size,
                mask_spatial_granularity=mask_spatial_granularity[i]
            )

            self.add_module(f"block{stage_index}-{i}", block)
        
    def forward(self, x, temperature):
        for block in self:
            x = block(x, temperature)
        return x


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
        
        input_size=224,
        mask_channel_group=[1],
        mask_spatial_granularity=[1],
        lr_mult=None
    ) -> None:
        super().__init__()

        assert lr_mult is not None
        self.lr_mult = lr_mult
        
        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        output_sizes = [input_size//4, input_size//8, input_size//16, input_size//32]
        depth_stages = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            depth_stages.append(depth)

        self.trunk_output = nn.ModuleList()
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            if i==0:
                mask_channel_group_stage = mask_channel_group[:depth_stages[0]]
                mask_spatial_granularity_stage = mask_spatial_granularity[:depth_stages[0]]
                
                _prev_blocks = depth_stages[0]
            else:
                _prev_blocks = sum(depth_stages[:i])
                mask_channel_group_stage = mask_channel_group[_prev_blocks:_prev_blocks+depth_stages[i]]
                mask_spatial_granularity_stage = mask_spatial_granularity[_prev_blocks:_prev_blocks+depth_stages[i]]
            self.trunk_output.add_module(
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                        mask_channel_group=mask_channel_group_stage,
                        output_size=output_sizes[i],
                        mask_spatial_granularity=mask_spatial_granularity_stage
                    ),
            )

            current_width = width_out

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, temperature:float) -> Tensor:
        c_in = x.shape[1]
        x = self.stem(x)
        b,c,h,w = x.shape
        flops = c_in * c * h * w * 9
        
        sparsity_list = None
        flops_perc_list = None
        sparsity_list_dil = None
        
        x = (x, sparsity_list, sparsity_list_dil, flops_perc_list, flops)
        for stage in self.trunk_output:
            x = stage(x, temperature)
        
        x, sparsity_list, sparsity_list_dil, flops_perc_list, flops = x
        
        x = self.avgpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]
        
        x = x.flatten(start_dim=1)
        c_in = x.shape[1]
        x = self.fc(x)
        flops += c_in*x.shape[1]

        return x, sparsity_list, sparsity_list_dil, flops_perc_list, flops
    
    def get_optim_policies(self):
        backbone_params = []
        masker_params = []

        for name, m in self.named_modules():
            if 'masker' in name:
                if isinstance(m, torch.nn.Conv2d):
                    ps = list(m.parameters())
                    masker_params.append(ps[0]) # ps[0] is a tensor, use append
                    if len(ps) == 2:
                        masker_params.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    masker_params.extend(list(m.parameters()))  # this is a list, use extend
            else:
                if isinstance(m, torch.nn.Conv2d):
                    ps = list(m.parameters())
                    backbone_params.append(ps[0]) # ps[0] is a tensor, use append
                    if len(ps) == 2:
                        backbone_params.append(ps[1])
                elif isinstance(m, torch.nn.BatchNorm2d):
                    backbone_params.extend(list(m.parameters()))  # this is a list, use extend
                elif isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    backbone_params.append(ps[0])
                    if len(ps) == 2:
                        backbone_params.append(ps[1])
        return [
            {'params': backbone_params, 'lr_mult': self.lr_mult, 'decay_mult': 1.0, 'name': "backbone_params"},
            {'params': masker_params, 'lr_mult': 1.0, 'decay_mult': 1.0, 'name': "masker_params"},
        ]


def _regnet(arch: str, block_params: BlockParams, pretrained: bool, progress: bool, **kwargs: Any) -> RegNet:
    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)
    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def dyn_regnet_y_400mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_400MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs)
    return _regnet("regnet_y_400mf", params, pretrained, progress, **kwargs)


def dyn_regnet_y_800mf(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RegNet:
    """
    Constructs a RegNetY_800MF architecture from
    `"Designing Network Design Spaces" <https://arxiv.org/abs/2003.13678>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25, **kwargs)
    return _regnet("regnet_y_800mf", params, pretrained, progress, **kwargs)
