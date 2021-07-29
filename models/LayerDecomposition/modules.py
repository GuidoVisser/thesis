import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from InputProcessing.FlowHandler.flowHandler import FlowHandler

class ConvBlock(nn.Module):
    """Helper module consisting of a convolution, optional normalization and activation, with padding='same'.
    
    Adapted from Layered Neural Rendering (https://github.com/google/retiming)
    """

    def __init__(self, conv, in_channels, out_channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Create a conv block.

        Parameters:
            conv (convolutional layer) - - the type of conv layer, e.g. Conv2d, ConvTranspose2d
            in_channels (int) - - the number of input channels
            in_channels (int) - - the number of output channels
            ksize (int) - - the kernel size
            stride (int) - - stride
            dil (int) - - dilation
            norm (norm layer) - - the type of normalization layer, e.g. BatchNorm2d, InstanceNorm2d
            activation (str)  -- the type of activation: relu | leaky | tanh | none
        """
        super(ConvBlock, self).__init__()
        self.k = ksize
        self.s = stride
        self.d = dil
        self.conv = conv(in_channels, out_channels, ksize, stride=stride, dilation=dil)

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        height, width = x.shape[-2:]
        if isinstance(self.conv, nn.modules.ConvTranspose2d):
            desired_height = height * self.s
            desired_width = width * self.s
            pady = 0
            padx = 0
        else:
            desired_height = height // self.s
            desired_width = width // self.s
            pady = .5 * (self.s * (desired_height - 1) + (self.k - 1) * (self.d - 1) + self.k - height)
            padx = .5 * (self.s * (desired_width - 1) + (self.k - 1) * (self.d - 1) + self.k - width)
        x = F.pad(x, [int(np.floor(padx)), int(np.ceil(padx)), int(np.floor(pady)), int(np.ceil(pady))])
        x = self.conv(x)
        if x.shape[-2] != desired_height or x.shape[-1] != desired_width:
            cropy = x.shape[-2] - desired_height
            cropx = x.shape[-1] - desired_width
            x = x[:, :, int(np.floor(cropy / 2.)):-int(np.ceil(cropy / 2.)),
                int(np.floor(cropx / 2.)):-int(np.ceil(cropx / 2.))]
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LayerDecompositionUNet(nn.Module):
    def __init__(self, conv_channels=64, in_channels=16, max_frames=200, coarseness=10):
        super().__init__()
        self.encoder = nn.ModuleList([
            ConvBlock(nn.Conv2d, in_channels, conv_channels, ksize=4, stride=2),
            ConvBlock(nn.Conv2d, conv_channels, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')])
        self.decoder = nn.ModuleList([
            ConvBlock(nn.ConvTranspose2d, conv_channels * 4 * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, conv_channels * 4 * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, conv_channels * 4 * 2, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, conv_channels * 2 * 2, conv_channels, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, conv_channels * 2, conv_channels, ksize=4, stride=2, norm=nn.BatchNorm2d)])
        self.final_rgba = ConvBlock(nn.Conv2d, conv_channels, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock(nn.Conv2d, conv_channels, 2, ksize=4, stride=1, activation='none')

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


    def forward(self, input, background_flow):
        """
        Apply forward pass of network
        """

        batch_size, N_t, N_layers, channels, H, W = input.shape

        input_t0 = input[:, 0]
        input_t1 = input[:, 1]

        composite_rgba = None
        composite_flow = background_flow

        layers_rgba = []
        layers_flow = []

        # For temporal consistency
        layers_alpha_warped = []
        composite_warped = None

        # TODO camera stabilization correction

        for i in range(N_layers):
            layer_input = torch.cat(([input_t0[:, i], input_t1[:, i]]))

            rgba, flow, _ = self.render(layer_input)
            alpha = rgba[:, 3:4] * .5 + 5

            if i == 0:
                composite_rgba = rgba
                flow = background_flow
                
                # Temporal consistency 
                rgba_warped = rgba[:batch_size]
                composite_warped = rgba_warped[:, :3]

            else:
                composite_rgba = rgba * alpha + composite_rgba * (1. - alpha)
                composite_flow = flow * alpha + composite_flow * (1. - alpha)

                # Temporal consistency
                rgba_t1 = rgba[batch_size:]
                rgba_warped = FlowHandler.apply_flow(rgba_t1, flow[:batch_size])
                alpha_warped = rgba_warped[:, 3:4] * .5 + .5
                composite_warped = rgba_warped[:, :3] * alpha_warped + composite_warped * (1.0 - alpha_warped)

            layers_rgba.append(rgba)
            layers_flow.append(flow)
            layers_alpha_warped.append(rgba_warped[:, 3:4])

        # stack t, t+1 channelwise
        composite_rgba = torch.stack((composite_rgba[:batch_size], composite_rgba[batch_size:]), 1)
        composite_flow = torch.stack((composite_flow[:batch_size], composite_flow[batch_size:]), 1)
        layers_rgba = torch.stack(layers_rgba, 2)
        layers_rgba = torch.stack((layers_rgba[:batch_size], layers_rgba[batch_size:]), 1)
        layers_flow = torch.stack(layers_flow, 2)
        layers_flow = torch.stack((layers_flow[:batch_size], layers_flow[batch_size:]), 1)

        out = {
            "rgba_reconstruction": composite_rgba,                      # [B, 2, 4, H, W]
            "flow_reconstruction": composite_flow,                      # [B, 2, 2, H, w]
            "reconstruction_warped": composite_warped,                  # [B, 2, 4, H, W]
            "layers_rgba": layers_rgba,                                 # [B, 2, L, 4, H, W]
            "layers_flow": layers_flow,                                 # [B, 2, L, 2, H, W]
            "layers_alpha_warped": torch.stack(layers_alpha_warped, 2)  # [B, L, 1, H, W]
        }
        return out


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

        # get camera adjustment params
        bg_offset = F.interpolate(self.bg_offset, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        bg_offset = bg_offset[0, :, index].transpose(0, 1)
        bg_offset = F.grid_sample(bg_offset, jitter.permute(0, 2, 3, 1), align_corners=True)
        br_scale = F.interpolate(self.brightness_scale, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        br_scale = br_scale[0, 0, index].unsqueeze(1)
        br_scale = F.grid_sample(br_scale, jitter.permute(0, 2, 3, 1), align_corners=True)

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
                    rgba = FlowHandler.apply_flow(rgba, bg_offset)
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
                rgba_warped = FlowHandler.apply_flow(rgba_t1, flow[:b_sz])
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
        composite_rgb = torch.cat((composite_rgb[:b_sz], composite_rgb[b_sz:]), 1)
        composite_flow = torch.cat((composite_flow[:b_sz], composite_flow[b_sz:]), 1)
        layers_rgba = torch.stack(layers_rgba, 2)
        layers_rgba = torch.cat((layers_rgba[:b_sz], layers_rgba[b_sz:]), 1)
        layers_flow = torch.stack(layers_flow, 2)
        layers_flow = torch.cat((layers_flow[:b_sz], layers_flow[b_sz:]), 1)
        br_scale = torch.cat((br_scale[:b_sz], br_scale[b_sz:]), 1)
        bg_offset = torch.cat((bg_offset[:b_sz], bg_offset[b_sz:]), 1)

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

        # get camera adjustment params
        bg_offset = F.interpolate(self.bg_offset, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        bg_offset = bg_offset[0, :, index].transpose(0, 1)
        bg_offset = F.grid_sample(bg_offset, jitter.permute(0, 2, 3, 1), align_corners=True)
        br_scale = F.interpolate(self.brightness_scale, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        br_scale = br_scale[0, 0, index].unsqueeze(1)
        br_scale = F.grid_sample(br_scale, jitter.permute(0, 2, 3, 1), align_corners=True)

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
        composite_rgb = torch.cat((composite_rgb[:b_sz], composite_rgb[b_sz:]), 1)
        composite_flow = torch.cat((composite_flow[:b_sz], composite_flow[b_sz:]), 1)
        layers_rgba = torch.stack(layers_rgba, 2)
        layers_rgba = torch.cat((layers_rgba[:b_sz], layers_rgba[b_sz:]), 1)
        layers_flow = torch.stack(layers_flow, 2)
        layers_flow = torch.cat((layers_flow[:b_sz], layers_flow[b_sz:]), 1)
        br_scale = torch.cat((br_scale[:b_sz], br_scale[b_sz:]), 1)
        bg_offset = torch.cat((bg_offset[:b_sz], bg_offset[b_sz:]), 1)

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