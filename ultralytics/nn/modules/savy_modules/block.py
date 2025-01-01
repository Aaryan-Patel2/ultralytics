import torch.nn as nn
import torch
import torch.functional as F
from adaptive_convolution.conv_runner import AdaptiveConv
from kornia.filters import gaussian_blur2d
from typing import Tuple



class GlobalPoolingAggregator(nn.Module):
    def __init__(self, in_channels: int, spatial_dim: Tuple[int, int], c_final: int):
        super(GlobalPoolingAggregator, self).__init__()
        self.c_final = c_final
        self.spatial_dim = spatial_dim

        self.branch1_conv = nn.Conv2d(in_channels=in_channels, out_channels=c_final, kernel_size=1)
        self.branch2_conv = nn.Conv2d(in_channels=in_channels, out_channels=c_final, kernel_size=1)
        self.branch3_conv = nn.Conv2d(in_channels=in_channels, out_channels=c_final, kernel_size=1)

    def forward(self, x1, x2, x3):
        # Apply 1x1 Convolutions (seperate) to each tensor to resize to exact same dimensions

        # Bilinear interpolation to target spatial dimensions
        branch1 = F.interpolate(x1, size=self.spatial_dim, mode='bilinear', align_corners=False)
        branch2 = F.interpolate(x2, size=self.spatial_dim, mode='bilinear', align_corners=False)
        branch3 = F.interpolate(x3, size=self.spatial_dim, mode='bilinear', align_corners=False)

        # Apply separate pointwise convolutions to each branch
        branch1 = self.branch1_conv(branch1)
        branch2 = self.branch2_conv(branch2)
        branch3 = self.branch3_conv(branch3)

        # Get the average for a single map representation
        map = (branch1 + branch2 + branch3) / 3

        return map

    """
    Generalized Joint Bilateral Upsampler (JBU) Kernels
    https://github.com/mhamilton723/FeatUp/blob/main/featup/upsamplers.py
    """


class JBULearnedRange(torch.nn.Module):
    """
    JBULearnedRange module for upsampling using generalized spatial and range kernels.


    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.


    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.


    Methods:
        forward: Applies the range and spatial kernels to the low resolution features with normalization
        get_range_kernel: Utilizes the source (guidance) feature map to compute a temperature weighted softmax
        that  integrates information from the guidance pixels and feature map pixels (w.r.t the guidance image)
        get_spatial_kernel: Utilizes a learned gaussian kernel + Euclidean distance to find the spatial simil-
        arities betwen the high-res (guidance) pixels and the feature map pixels within the distance grid
   
    Examples:
        >>> import torch
        >>> from ultralytics import JBULearnedRange
        >>> model = SCDown(guidance_dim=, feat_dim=,key_dim=, scale=2, radius=3)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """
    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))


    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded) \
            .reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)) \
            .permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self, device):
        dist_range = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel(source.device)
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

        # (B C, H+Pad, W+Pad) x (B, H, W, KH, KW) -> BCHW; Figure out CUDA impl.
        result =  AdaptiveConv.apply(hr_source_padded, combined_kernel)
        return None

class JBUStack(torch.nn.Module):
    def __init__(self, feat_dim, stack_count, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsamplers = torch.nn.ModuleList(
            [JBULearnedRange(3, feat_dim, 32, radius=3) for _ in range(self.stack_count)]
        )
        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1)
        )
    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled




    def forward(self, source, guidance):
        for up in self.upsamplers:
            source = self.upsample(source, guidance, up)
        return self.fixup_proj(source) * 0.1 + source




class AttentionDownsampler(torch.nn.Module):

    def __init__(self, dim, kernel_size, final_size, blur_attn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.in_dim = dim
        self.attention_net = torch.nn.Sequential(
            torch.nn.Dropout(p=.2),
            torch.nn.Linear(self.in_dim, 1)
        )
        self.w = torch.nn.Parameter(torch.ones(kernel_size, kernel_size).cuda()
                                    + .01 * torch.randn(kernel_size, kernel_size).cuda())
        self.b = torch.nn.Parameter(torch.zeros(kernel_size, kernel_size).cuda()
                                    + .01 * torch.randn(kernel_size, kernel_size).cuda())
        self.blur_attn = blur_attn

    def forward_attention(self, feats, guidance):
        return self.attention_net(feats.permute(0, 2, 3, 1)).squeeze(-1).unsqueeze(1)

    def forward(self, hr_feats, guidance):
        b, c, h, w = hr_feats.shape
        if self.blur_attn:
            inputs = gaussian_blur2d(hr_feats, 5, (1.0, 1.0))
        else:
            inputs = hr_feats

        stride = (h - self.kernel_size) // (self.final_size - 1)
        patches = torch.nn.Unfold(self.kernel_size, stride=stride)(inputs) \
            .reshape(
            (b, self.in_dim, self.kernel_size * self.kernel_size, self.final_size, self.final_size * int(w / h))) \
            .permute(0, 3, 4, 2, 1)
        patch_logits = self.attention_net(patches).squeeze(-1)

        b, h, w, p = patch_logits.shape
        dropout = torch.rand(b, h, w, 1, device=patch_logits.device) > 0.2

        w = self.w.flatten().reshape(1, 1, 1, -1)
        b = self.b.flatten().reshape(1, 1, 1, -1)


        patch_attn_logits = (patch_logits * dropout) * w + b
        patch_attention = F.softmax(patch_attn_logits, dim=-1)
        downsampled = torch.einsum("bhwpc,bhwp->bchw", patches, patch_attention)

        return downsampled[:, :c, :, :]