import torch
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .spynet_arch import SpyNet
from .unet_arch import UNet
from .arch_util import SizeAdapter


@ARCH_REGISTRY.register()
class EBVSR(nn.Module):
    """An Event-driven Bidirectional Video Super-Resolution (EBVSR) network for 4x VSR

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, voxel_bins=5, spynet_path=None, unet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)
        self.unet = UNet(inChannels=voxel_bins, outChannels=2, load_path=unet_path)

        # fusion
        self.bmcs = BMCS(num_feat, voxel_bins)

        # backward
        self.backward_conv1 = ConvResidualBlocks(num_feat, num_feat, num_block)
        self.backward_conv2 = ConvResidualBlocks(num_feat, num_feat, num_block)
        self.conv_b = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.fusion_b = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        #forward
        self.forward_conv1 = ConvResidualBlocks(num_feat, num_feat, num_block)
        self.forward_conv2 = ConvResidualBlocks(num_feat, num_feat, num_block)
        self.conv_f = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.fusion_f = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.upconv3 = nn.Conv2d(3, 3 * 4, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(3, 3 * 4, 3, 1, 1, bias=True)
        self.conv_last2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv_last3 = nn.Conv2d(3 * 2, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_image_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        img_flow_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        img_flow_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return img_flow_forward, img_flow_backward

    def get_voxel_flow(self, voxels_f, voxels_b):

        b, n, c, h, w = voxels_f.size()

        v_f = voxels_b.reshape(-1, c, h, w)
        v_b = voxels_f.reshape(-1, c, h, w)

        voxel_flow_backward = self.unet(v_b).view(b, n, 2, h, w)
        voxel_flow_forward = self.unet(v_f).view(b, n, 2, h, w)

        return voxel_flow_forward, voxel_flow_backward

    def forward(self, imgs, voxels_f, voxels_b):
        """Forward function of EBVSR 4x VSR

        Args:
            imgs: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
            voxels_f: forward events with shape (b, n-1 , c, h, w).
            voxels_b: backward events with shape (b, n-1 , c, h, w).

        Output:
            out_l: output frames with shape (b, n, c, h, w)
        """
        img_flow_forward, img_flow_backward = self.get_image_flow(imgs)
        voxel_flow_forward, voxel_flow_backward = self.get_voxel_flow(voxels_f, voxels_b)

        b, n, c, h, w = imgs.size()

        # backward branch
        out_l_b = []
        feat_prop = imgs.new_zeros(b, self.num_feat, h, w)
        feat_prop_img = torch.zeros_like(feat_prop)
        feat_prop_voxel = torch.zeros_like(feat_prop)
        for i in range(n - 1, -1, -1):
            x_i = imgs[:, i, :, :, :]
            if i < n - 1:
                img_flow = img_flow_backward[:, i, :, :, :]
                voxel_flow = voxel_flow_backward[:, i, :, :, :]
                feat_prop_img = flow_warp(feat_prop, img_flow.permute(0, 2, 3, 1))
                feat_prop_voxel = flow_warp(feat_prop, voxel_flow.permute(0, 2, 3, 1))

            feat_prop_img = self.backward_conv1(feat_prop_img)
            feat_prop_voxel = self.backward_conv2(feat_prop_voxel)
            feat_prop = self.conv_b(torch.cat([feat_prop_img, feat_prop_voxel], dim=1))
            feat_prop = self.fusion_b(torch.cat([x_i, feat_prop], dim=1))

            out_l_b.insert(0, feat_prop)

        # forward branch
        out_l_f = []
        feat_prop = torch.zeros_like(feat_prop)
        feat_prop_img = torch.zeros_like(feat_prop)
        feat_prop_voxel = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = imgs[:, i, :, :, :]
            if i > 0:
                img_flow = img_flow_forward[:, i - 1, :, :, :]
                voxel_flow = voxel_flow_forward[:, i - 1, :, :, :]
                feat_prop_img = flow_warp(feat_prop, img_flow.permute(0, 2, 3, 1))
                feat_prop_voxel = flow_warp(feat_prop, voxel_flow.permute(0, 2, 3, 1))

            feat_prop_img = self.forward_conv1(feat_prop_img)
            feat_prop_voxel = self.forward_conv2(feat_prop_voxel)
            feat_prop = self.conv_f(torch.cat([feat_prop_img, feat_prop_voxel], dim=1))
            feat_prop = self.fusion_f(torch.cat([x_i, feat_prop], dim=1))

            out_l_f.append(feat_prop)

        # upsample
        assert len(out_l_b) == len(out_l_f) == n
        result = []
        template = voxels_f[:, 0, :, :, :]
        for i in range(0, n):
            # recurrent result
            x_i = imgs[:, i, :, :, :]
            out = torch.cat([out_l_b[i], out_l_f[i]], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            # synthesis result
            if i < n - 1:
                v_next = voxels_b[:, i, :, :, :]
            else:
                v_next = torch.zeros_like(template)
            if i == 0:
                v_pre = torch.zeros_like(template)
            else:
                v_pre = voxels_f[:, i - 1, :, :, :]
            output = self.bmcs(v_pre, x_i, v_next)
            output = self.lrelu(self.pixel_shuffle(self.upconv3(output)))
            output = self.lrelu(self.pixel_shuffle(self.upconv4(output)))
            output = self.conv_last2(output)

            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            output = torch.cat((out, output), dim=1)
            output = self.conv_last3(output)
            output = base + out
            result.append(output)

        return torch.stack(result, dim=1)


class BMCS(nn.Module):
    """Bidirectional, Multi-scale Cross-modality Synthesis module (BMCS).

    Args:
        num_feat (int): Number of channels. Default: 64.
        voxel_bins (int): Number of voxel bins. Default: 5.
        img_in_ch (int): Number of img channels. Default: 3.
    """

    def __init__(self, num_feat=64, voxel_bins=5, img_in_ch=3):
        super().__init__()
        self.num_feat = num_feat
        self.e_extractor = Feat_Exactor(voxel_bins)
        self.i_extractor = Feat_Exactor(img_in_ch)
        self.down = down(96, 128, 3)
        self.up = nn.ModuleList([up(128, 96), up(96, 64), up(64, 32), up(32, 16), up(16, 8)])
        self.bct = nn.ModuleList([BCT(96), BCT(64), BCT(32), BCT(16), BCT(8)])
        self.conv_last = nn.Conv2d(8, 3, 3, 1, 1)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, E1, I_mid, E2):
        """
        Args:
            E1, E2 (Tensor): previous time and posterior time voxels. (B, Bins, H, W)
            I_mid (Tensor): intermediate timestamp frame. (B, C, H, W)

        Output:
            Feat_fused (Tensor): feat fused among E1, E2 and I_mid. (B, num_feat, H, W)
        """
        b, c, h, w = I_mid.size()
        assert h >= 32 and w >= 32, 'we need input size must larger than 32*32'
        E1_pyramid, _ = self.e_extractor(E1)
        E2_pyramid, _ = self.e_extractor(E2)
        I_pyramid, (pad_h, pad_w) = self.i_extractor(I_mid)

        for s in range(len(I_pyramid)):
            F_pre = E1_pyramid[s]
            F_t = I_pyramid[s]
            F_post = E2_pyramid[s]
            f_agg = self.bct[s](F_pre, F_t, F_post)
            if s == 0:
                out = self.down(F_t)
            out = self.up[s](out, f_agg)
        # unpad out
        out = self.lrelu(self.conv_last(out))
        out = out[..., pad_h:, pad_w:]
        return out


class BCT(nn.Module):
    """Bidirectional Cross-modality Transformer (BCT)

    Args:
        dim (int): feature channel dimmension c.
    """

    def __init__(self, dim):
        super().__init__()
        self.cam = CAM_Module(dim)
        self.esam = ESAM_Module(dim)

        # fusion
        self.conv_cam = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True), nn.BatchNorm2d(dim))
        self.conv_esam = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True), nn.BatchNorm2d(dim))
        self.fusion = nn.Conv2d(dim*2, dim, 1, 1, 0, bias=True)
        self.gama = nn.Parameter(torch.ones(1))

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, F_pre, F_t, F_post):
        """
        Args:
            F_pre (Tensor): s-th scale previous voxel feature. [B, C, H, W]
            F_t (Tensor): s-th scale image feature. [B, C, H, W]
            F_post (Tensor): s-th scale posterior voxel feature. [B, C, H, W]

        Output:
            F_Agg (Tensor): fused feature by bidirectional cross attention at s-th scale. [B, C, H, W]
        """
        identity = F_t
        feat_channel = self.cam(F_t, F_pre)
        feat_spatial = self.esam(F_t, F_post)
        feat_channel = self.conv_cam(feat_channel)
        feat_spatial = self.conv_esam(feat_spatial)
        out = self.lrelu(self.fusion(torch.cat([feat_channel, feat_spatial], dim=1)))
        return self.gama * identity + out

class CAM_Module(nn.Module):
    """ Channel Attn Module

    Args:
        dim (int): feature channel dimmension c.
    """

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.k_proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.v_proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        """
        Args:
            x (Tensor): s-th scale img feature. [B, C, H, W]
            y (Tensor): s-th scale previous voxel feature. [B, C, H, W]

        Output:
            out (Tensor): channel attention to img feature result. [B, C, H, W]
        """
        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = x.shape

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = rearrange(q, 'b c h w -> b (h w) c') # image, [B, N, C]
        k = rearrange(k, 'b c h w -> b (h w) c') # event, [B, N, C]
        v = rearrange(v, 'b c h w -> b (h w) c') # event, [B, N, C]

        q = F.normalize(q, dim=-1)  # [B, N, C]
        k = F.normalize(k, dim=-1)  # [B, N, C]

        attn = torch.bmm(k.transpose(-2, -1), q).softmax(dim=-1)  # [B, C, C]
        out = torch.bmm(v, attn)  # [B, N, C]
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)

        out = self.alpha * out + x  # [B, C, H, W]

        return out


class ESAM_Module(nn.Module):
    """ Efficient Spatial Attn Module

    Args:
        dim (int): feature channel dimmension c.
    """

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.k_proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.v_proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        """
        Args:
            x (Tensor): s-th scale img feature. [B, C, H, W]
            y (Tensor): s-th scale posterior voxel feature. [B, C, H, W]

        Output:
            out (Tensor): spatial attention to img feature result. [B, C, H, W]
        """
        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b, c, h, w = x.shape

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = rearrange(q, 'b c h w -> b (h w) c') # image, [B, N, C]
        k = rearrange(k, 'b c h w -> b (h w) c') # event, [B, N, C]
        v = rearrange(v, 'b c h w -> b (h w) c') # event, [B, N, C]

        q = F.normalize(q, dim=-2)  # [B, N, C]
        k = F.normalize(k, dim=-2)  # [B, N, C]

        attn = torch.bmm(k.transpose(-2, -1), v).softmax(dim=-1)  # [B, C, C]
        out = torch.bmm(q, attn)  # [B, N, C]
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w) # [B, C, H, W]

        out = self.beta * out + x  # [B, C, H, W]

        return out


class up(nn.Module):

    def __init__(self, inChannels, outChannels):
        super(up, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x

class down(nn.Module):

    def __init__(self, inChannels, outChannels, filterSize):
        super(down, self).__init__()
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = F.leaky_relu(self.bn(self.conv1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn(self.conv2(x)), negative_slope=0.1)
        return x


class Feat_Exactor(nn.Module):
    """Feature Exactor to construct feature pyramid

    Args:
        num_in_ch (int): Number of input channels. Default: 3.

    """

    def __init__(self, num_in_ch=3):
        super().__init__()
        self._size_adapter = SizeAdapter(minimum_size=32)
        self.conv1 = nn.Conv2d(num_in_ch, 8, 7, 1, 3)
        self.down1 = down(8, 16, 5)
        self.down2 = down(16, 32, 3)
        self.down3 = down(32, 64, 3)
        self.down4 = down(64, 96, 3)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape is (B, C, H, W)

        Output:
            set(s1, s2, s3, s4, s5): Multi-scale feature
            set(pad_height, pad_width): for unpad to input image
        """
        x = self._size_adapter.pad(x)
        pad_height, pad_width = self._size_adapter._pixels_pad_to_height, self._size_adapter._pixels_pad_to_width
        s0 = F.leaky_relu(self.conv1(x), negative_slope=0.1) # [B, 8, 64, 64]
        s1 = self.down1(s0)  # [B, 16, 32, 32]
        s2 = self.down2(s1)  # [B, 32, 16, 16]
        s3 = self.down3(s2)  # [B, 64, 8, 8]
        s4 = self.down4(s3)  # [B, 96, 4, 4]
        return (s4, s3, s2, s1, s0), (pad_height, pad_width)



class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)