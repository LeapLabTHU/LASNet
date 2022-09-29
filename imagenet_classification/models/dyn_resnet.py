import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F

__all__ = ['dyn_resnet50', 'dyn_resnet101']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


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
        self.conv2.bias.data[:mask_channel_group] = 5.0
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


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None,
                 mask_channel_group=1,
                 output_size=56,
                 mask_spatial_granularity=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        base_width = 64
        width = int(planes * (base_width / 64.)) * group_width
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, group_width, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.conv1_flops_per_pixel = inplanes*width
        self.conv2_flops_per_pixel = width*width*9 // self.conv2.groups
        self.conv3_flops_per_pixel = width*planes*self.expansion

        if self.downsample is not None:
            self.downsample_flops = inplanes * planes * self.expansion

        self.output_size = output_size
        self.mask_spatial_granularity = mask_spatial_granularity
        self.mask_size = self.output_size // self.mask_spatial_granularity
        self.masker = Masker(inplanes, mask_channel_group, self.mask_size, feature_size=output_size, dilate_stride=stride)

    def forward(self, x, temperature=1.0):
        
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
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        dense_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv1_flops_per_pixel * out.shape[2] * out.shape[3] * sparsity_dil
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        dense_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv2_flops_per_pixel * out.shape[2] * out.shape[3] * sparsity
        
        out = self.conv3(out)
        out = self.bn3(out)
        dense_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3]
        sparse_flops += self.conv3_flops_per_pixel * out.shape[2] * out.shape[3] * sparsity
        
        out = apply_mask(out, mask)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            dense_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]
            sparse_flops += self.downsample_flops * identity.shape[2] * identity.shape[3]
        
        out += identity
        out = self.relu(out)

        flops += sparse_flops
        perc = sparse_flops / dense_flops

        if flops_perc_list == None:
            flops_perc_list = perc.unsqueeze(0)
        else:
            flops_perc_list = torch.cat((flops_perc_list,perc.unsqueeze(0)),dim=0)
        
        return out, sparsity_list, sparsity_list_dil, flops_perc_list, flops


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, width_mult=1.,
                 input_size=224,
                 mask_channel_group=[1],
                 mask_spatial_granularity=[1],
                 lr_mult=None, **kwargs):
        super(ResNet, self).__init__()

        assert lr_mult is not None
        self.lr_mult = lr_mult

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64*width_mult)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, int(64*width_mult), layers[0], stride=1,
                                       dilate=False,
                                       output_size=input_size//4,
                                       mask_channel_group=mask_channel_group[:layers[0]],
                                       mask_spatial_granularity=mask_spatial_granularity[:layers[0]])
        
        self.layer2 = self._make_layer(block, int(128*width_mult), layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       output_size=input_size//8,
                                       mask_channel_group=mask_channel_group[layers[0]:layers[0]+layers[1]],
                                       mask_spatial_granularity=mask_spatial_granularity[layers[0]:layers[0]+layers[1]])
        
        self.layer3 = self._make_layer(block, int(256*width_mult), layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       output_size=input_size//16,
                                       mask_channel_group=mask_channel_group[layers[0]+layers[1]:layers[0]+layers[1]+layers[2]],
                                       mask_spatial_granularity=mask_spatial_granularity[layers[0]+layers[1]:layers[0]+layers[1]+layers[2]])
        
        self.layer4 = self._make_layer(block, int(512*width_mult), layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       output_size=input_size//32,
                                       mask_channel_group=mask_channel_group[layers[0]+layers[1]+layers[2]:],
                                       mask_spatial_granularity=mask_spatial_granularity[layers[0]+layers[1]+layers[2]:])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512*width_mult * block.expansion), num_classes)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'masker' not in name:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    output_size=56,
                    mask_channel_group=[1],
                    mask_spatial_granularity=[1]):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, group_width=self.groups,
                            dilation=previous_dilation, norm_layer=norm_layer, 
                            output_size=output_size,
                            mask_channel_group=mask_channel_group[0],
                            mask_spatial_granularity=mask_spatial_granularity[0]))
        self.inplanes = planes * block.expansion
        for j in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_width=self.groups,
                                dilation=self.dilation,
                                norm_layer=norm_layer, 
                                output_size=output_size,
                                mask_channel_group=mask_channel_group[j],
                                mask_spatial_granularity=mask_spatial_granularity[j]))

        return nn.ModuleList(layers)

    def forward(self, x, temperature):
        c_in = x.shape[1]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        flops = c_in * x.shape[1] * x.shape[2] * x.shape[3] * self.conv1.weight.shape[2]*self.conv1.weight.shape[3]

        x = self.maxpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]*9
        
        sparsity_list = None
        flops_perc_list = None
        sparsity_list_dil = None
        
        x = (x, sparsity_list, sparsity_list_dil, flops_perc_list, flops)
        for i in range(len(self.layer1)):
            x = self.layer1[i](x, temperature)
        
        for i in range(len(self.layer2)):
            x = self.layer2[i](x, temperature)

        for i in range(len(self.layer3)):
            x = self.layer3[i](x, temperature)

        for i in range(len(self.layer4)):
            x = self.layer4[i](x, temperature)
        
        x, sparsity_list, sparsity_list_dil, flops_perc_list, flops = x
        
        x = self.avgpool(x)
        flops += x.shape[1]*x.shape[2]*x.shape[3]
        
        x = torch.flatten(x, 1)
        
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


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def dyn_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 50')
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def dyn_resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    print('Model: Resnet 101')
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)
