# Copyright 2021 Erika Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from ..third_party.models.networks import init_net
    from ..third_party.models.networks_lnr import ConvBlock
except:
    from third_party.models.networks import init_net
    from third_party.models.networks_lnr import ConvBlock

import cv2
import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)
###############################################################################
# Helper Functions
###############################################################################
def define_omnimatte(nf=64, in_c=16, gpu_ids=[]):
    """Create a layered neural renderer.

    Parameters:
        nf (int) -- the number of channels in the first/last conv layers
        in_c (int) -- the number of input channels
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a layered neural rendering model.
    """
    net = Omnimatte(nf, in_c)
    return init_net(net, gpu_ids)


##############################################################################
# Classes
##############################################################################
class Omnimatte(nn.Module):
    """Omnimatte model for video decomposition.

    Consists of UNet.
    """

    def __init__(self, nf=64, in_c=16, max_frames=200, coarseness=10):
        super(Omnimatte, self).__init__(),
        """Initialize Omnimatte model.

        Parameters:
            nf (int) -- the number of channels in the first/last conv layers
            in_c (int) -- the number of channels in the input
            max_frames (int) -- max number of frames in video
            coarseness (int) -- controls temporal dimension of camera adjustment params 
        """
        # Define UNet
        self.encoder = nn.ModuleList([
            ConvBlock(nn.Conv2d, in_c, nf, ksize=4, stride=2),
            ConvBlock(nn.Conv2d, nf, nf * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')])
        self.decoder = nn.ModuleList([
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 2, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 2 * 2, nf, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, nf * 2, nf, ksize=4, stride=2, norm=nn.BatchNorm2d)])
        self.final_rgba = ConvBlock(nn.Conv2d, nf, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock(nn.Conv2d, nf, 2, ksize=4, stride=1, activation='none')

        self.max_frames = max_frames
        self.bg_offset = nn.Parameter(torch.zeros(1, 2, max_frames // coarseness, 4, 7))
        self.brightness_scale = nn.Parameter(torch.ones(1, 1, max_frames // coarseness, 4, 7))

    def render(self, x):
        """Pass inputs for a single layer through UNet.

        Parameters:
            x (tensor) - - sampled texture concatenated with person IDs

        Returns RGBA for the input layer and the final feature maps.
        """
        skips = [x]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 5:
                skips.append(x)
        for layer in self.decoder:
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)
        rgba = self.final_rgba(x)
        flow = self.final_flow(x)
        return rgba, flow, x

    def forward(self, input, bg_flow, bg_warp, jitter, index, do_adj):
        """Forward pass through layered neural renderer.

        1. Split input to t and t+1 since they are concatenated channelwise
        2. Pass to UNet
        3. Composite RGBA outputs and flow outputs
        4. Warp alphas t+1 -> t using predicted flow layers
        5. Concat results t and t+1 channelwise (except warped alphas)

        Parameters:
            input (tensor) - - inputs for all layers, with shape [B, L, C*2, H, W]
            bg_flow (tensor) - - flow for background layer, with shape [B, 2*2, H, W]
            bg_warp (tensor) - - warping grid used to sample background layer from unwrapped background, with shape [B, 2*2, H, W]
            jitter (tensor) - - warping grid used to apply data transformation, with shape [B, 2*2, H, W]
            index (tensor) - - frame indices [B, 2]
            do_adj (bool) - - whether to apply camera adjustment parameters
        """
        b_sz, n_layers, channels, height, width = input.shape
        input_t = input[:, :, :channels // 2]
        input_t1 = input[:, :, channels // 2:]
        bg_warp = torch.cat((bg_warp[:, :2], bg_warp[:, 2:]))
        bg_flow = torch.cat((bg_flow[:, :2], bg_flow[:, 2:]))
        jitter = torch.cat((jitter[:, :2], jitter[:, 2:]))
        index = index.transpose(0, 1).reshape(-1)
        composite_rgb = None
        composite_flow = bg_flow
        layers_rgba = []
        layers_flow = []
        alphas_warped = []
        composite_warped = None
        # print(index[0])

        # get camera adjustment params
        bg_offset = F.interpolate(self.bg_offset, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        bg_offset = bg_offset[0, :, index].transpose(0, 1)
        bg_offset = F.grid_sample(bg_offset, jitter.permute(0, 2, 3, 1), align_corners=True)
        br_scale = F.interpolate(self.brightness_scale, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        br_scale = br_scale[0, 0, index].unsqueeze(1)
        br_scale = F.grid_sample(br_scale, jitter.permute(0, 2, 3, 1), align_corners=True)

        # debug0 = torch.clone(input_t).detach().cpu()
        # debug1 = torch.clone(input_t1).detach().cpu()
        # bg_uv = torch.clone(bg_warp).detach().cpu()
        # jit = torch.clone(jitter).detach().cpu()
        # bg_uv0 = bg_uv[0].permute(1, 2, 0).numpy()
        # bg_uv1 = bg_uv[1].permute(1, 2, 0).numpy()
        # jit0 = jit[0].permute(1, 2, 0).numpy()
        # jit1 = jit[1].permute(1, 2, 0).numpy()
        # b0 = debug0[0, 0].permute(1, 2, 0).numpy()
        # b1 = debug1[0, 0].permute(1, 2, 0).numpy()
        # l0 = debug0[0, 1].permute(1, 2, 0).numpy()
        # l1 = debug1[0, 1].permute(1, 2, 0).numpy()

        # if index[0].item() != 6000:        
        #     cv2.imshow("bg pid 0", b0[:, :, 0])
        #     cv2.imshow("bg pid 1", b1[:, :, 0])
        #     cv2.imshow("l pid 0",  l0[:, :, 0])
        #     cv2.imshow("l pid 1",  l1[:, :, 0])
        #     cv2.imshow("bg flow 0", flow_to_image(b0[:, :, 1:3]))
        #     cv2.imshow("bg flow 1", flow_to_image(b1[:, :, 1:3]))
        #     cv2.imshow("l flow 0",  flow_to_image(l0[:, :, 1:3]))
        #     cv2.imshow("l flow 1",  flow_to_image(l1[:, :, 1:3]))
        #     cv2.imshow("bg uv 0", flow_to_image(bg_uv0))
        #     cv2.imshow("bg uv 1", flow_to_image(bg_uv1))
        #     cv2.imshow("jitter 0",  flow_to_image(jit0))
        #     cv2.imshow("jitter 1",  flow_to_image(jit1))
        #     cv2.imshow("bg noise 0", b0[:, :, 3:6] * 0.5 + 0.5)
        #     cv2.imshow("bg noise 1", b1[:, :, 3:6] * 0.5 + 0.5)
        #     cv2.imshow("l noise 0",  l0[:, :, 3:6] * 0.5 + 0.5)
        #     cv2.imshow("l noise 1",  l1[:, :, 3:6] * 0.5 + 0.5)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


        for i in range(n_layers):
            # Get RGBA and flow for this layer.
            input_i = torch.cat((input_t[:, i], input_t1[:, i]))
            rgba, flow, last_feat = self.render(input_i)
            alpha = rgba[:, 3:4] * .5 + .5

            # Update the composite with this layer's RGBA output
            if i == 0:
                # sample from unwrapped background
                rgba = F.grid_sample(rgba, bg_warp.permute(0, 2, 3, 1), align_corners=True)
                # apply learned background offset
                if do_adj:
                    rgba = warp_flow(rgba, bg_offset.permute(0, 2, 3, 1))
                composite_rgb = rgba
                flow = bg_flow
                # for background layer, use prediction for t
                rgba_warped = rgba[:b_sz]
                composite_warped = rgba_warped[:, :3]
            else:
                composite_rgb = rgba * alpha + composite_rgb * (1.0 - alpha)
                composite_flow = flow * alpha + composite_flow * (1.0 - alpha)

                # warp rgba t+1 -> t and composite
                rgba_t1 = rgba[b_sz:]
                rgba_warped = warp_flow(rgba_t1, flow[:b_sz].permute(0, 2, 3, 1))
                alpha_warped = rgba_warped[:, 3:4] * .5 + .5
                composite_warped = rgba_warped[:, :3] * alpha_warped + composite_warped * (1.0 - alpha_warped)

            layers_rgba.append(rgba)
            layers_flow.append(flow)
            alphas_warped.append(rgba_warped[:, 3:4])

        if do_adj:
            # apply learned brightness scaling
            composite_rgb = br_scale * (composite_rgb * .5 + .5)
            composite_rgb = torch.clamp(composite_rgb, 0, 1)
            composite_rgb = composite_rgb * 2 - 1  # map back to [-1, 1] range
        # stack t, t+1 channelwise
        composite_rgb  = torch.cat((composite_rgb[:b_sz], composite_rgb[b_sz:]), 1)
        composite_flow = torch.cat((composite_flow[:b_sz], composite_flow[b_sz:]), 1)
        layers_rgba    = torch.stack(layers_rgba, 2)
        layers_rgba    = torch.cat((layers_rgba[:b_sz], layers_rgba[b_sz:]), 1)
        layers_flow    = torch.stack(layers_flow, 2)
        layers_flow    = torch.cat((layers_flow[:b_sz], layers_flow[b_sz:]), 1)
        br_scale       = torch.cat((br_scale[:b_sz], br_scale[b_sz:]), 1)
        bg_offset      = torch.cat((bg_offset[:b_sz], bg_offset[b_sz:]), 1)

        outputs = {
            'reconstruction_rgb': composite_rgb,
            'reconstruction_flow': composite_flow,
            'layers_rgba': layers_rgba,
            'layers_flow': layers_flow,
            'alpha_warped': torch.stack(alphas_warped, 2),
            'reconstruction_warped': composite_warped,
            'bg_offset': bg_offset,
            'brightness_scale': br_scale
        }
        return outputs


def warp_flow(tensor, flow):
    """Warp the tensor using the flow. Flow should be in pixel space."""
    h, w = tensor.shape[-2:]
    ramp_u = torch.arange(w).unsqueeze(0).repeat(h, 1)
    ramp_v = torch.arange(h).unsqueeze(-1).repeat(1, w)
    grid = torch.stack([ramp_u, ramp_v], -1).unsqueeze(0)
    grid = grid.float().to(flow.device) + flow
    grid[..., 0] /= (w - 1)
    grid[..., 1] /= (h - 1)
    grid = grid * 2 - 1
    warped = F.grid_sample(tensor, grid, align_corners=True)
    return warped
