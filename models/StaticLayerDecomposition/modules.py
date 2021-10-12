import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from InputProcessing.flowHandler import FlowHandler

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
    def __init__(self, conv_channels=64, in_channels=16, max_frames=200, coarseness=10, do_adjustment=True):
        super().__init__()
        self.encoder = nn.ModuleList([
            ConvBlock(nn.Conv2d, in_channels,       conv_channels,     ksize=4, stride=2),
            ConvBlock(nn.Conv2d, conv_channels,     conv_channels * 2, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 2, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=2,        norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')])
        
        self.decoder = nn.ModuleList([
            ConvBlock(nn.ConvTranspose2d, conv_channels * 4 * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, conv_channels * 4 * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, conv_channels * 4 * 2, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, conv_channels * 2 * 2, conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d),
            ConvBlock(nn.ConvTranspose2d, conv_channels * 2,     conv_channels,     ksize=4, stride=2, norm=nn.BatchNorm2d)])
        
        self.final_rgba = ConvBlock(nn.Conv2d, conv_channels, 4, ksize=4, stride=1, activation='tanh')
        self.final_flow = ConvBlock(nn.Conv2d, conv_channels, 2, ksize=4, stride=1, activation='none')

        self.bg_offset        = nn.Parameter(torch.zeros(1, 2, max_frames // coarseness, 4, 7))
        self.brightness_scale = nn.Parameter(torch.ones(1, 1, max_frames // coarseness, 4, 7))

        self.max_frames = max_frames
        self.do_adjustment = do_adjustment
        
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
        return rgba, flow


    def forward(self, input, disp=False):
        """
        Apply forward pass of network
        """

        input_tensor      = input["input_tensor"]
        background_flow   = input["background_flow"]
        background_uv_map = input["background_uv_map"]
        jitter_grid       = input["jitter_grid"]
        index             = input["index"]

        batch_size, N_t, N_layers, channels, H, W = input_tensor.shape

        input_t0 = input_tensor[:, 0]
        input_t1 = input_tensor[:, 1]

        composite_rgba = None
        composite_flow = torch.cat((background_flow[:, 0], background_flow[:, 1]))

        background_uv_map = torch.cat((background_uv_map[:, 0], background_uv_map[:, 1]))

        layers_rgba = []
        layers_flow = []

        # For temporal consistency
        layers_alpha_warped = []
        composite_warped = None

        # camera stabilization correction
        index = index.transpose(0, 1).reshape(-1)
        jitter_grid = torch.cat((jitter_grid[:, 0], jitter_grid[:, 1]))
        background_offset = self.get_background_offset(jitter_grid, index)
        brightness_scale  = self.get_brightness_scale(jitter_grid, index) 

        for i in range(N_layers):
            layer_input = torch.cat(([input_t0[:, i], input_t1[:, i]]))

            rgba, flow = self.render(layer_input)
            alpha = self.get_alpha_from_rgba(rgba)

            # Background layer
            if i == 0:
                rgba = F.grid_sample(rgba, background_uv_map)               
                if self.do_adjustment:
                    rgba = FlowHandler.apply_flow(rgba, background_offset)

                composite_rgba = rgba
                flow = composite_flow

                # Temporal consistency 
                rgba_warped      = rgba[:batch_size]
                composite_warped = rgba_warped[:, :3]
            # Object layers
            else:

                if disp:
                    composite_rgba_ = rgba * alpha + composite_rgba * (1. - alpha)
                    cv2.imshow("rgba", torch.clone(rgba[0]).detach().permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
                    cv2.imshow("composite", torch.clone(composite_rgba[0]).detach().permute(1, 2, 0).cpu().numpy()* 0.5 + 0.5)
                    cv2.imshow("alpha", torch.clone(alpha[0]).detach().permute(1, 2, 0).cpu().numpy()* 0.5 + 0.5)
                    cv2.imshow("composite_", torch.clone(composite_rgba_[0]).detach().permute(1, 2, 0).cpu().numpy()* 0.5 + 0.5)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                composite_rgba = rgba * alpha + composite_rgba * (1. - alpha)
                composite_flow = flow * alpha + composite_flow * (1. - alpha)

                # Temporal consistency
                rgba_t1          = rgba[batch_size:]
                rgba_warped      = FlowHandler.apply_flow(rgba_t1, flow[:batch_size])
                alpha_warped     = self.get_alpha_from_rgba(rgba_warped)
                composite_warped = rgba_warped[:, :3] * alpha_warped + composite_warped * (1.0 - alpha_warped)

            layers_rgba.append(rgba)
            layers_flow.append(flow)
            layers_alpha_warped.append(rgba_warped[:, 3:4])

        if self.do_adjustment:
            # map output to [0, 1]
            composite_rgba = composite_rgba * 0.5 + 0.5

            # adjust for brightness
            composite_rgba = torch.clamp(brightness_scale * composite_rgba, 0, 1)

            # map back to [-1, 1]
            composite_rgba = composite_rgba * 2 - 1

        # stack in time dimension
        composite_rgba      = torch.stack((composite_rgba[:batch_size], composite_rgba[batch_size:]), 1)
        composite_flow      = torch.stack((composite_flow[:batch_size], composite_flow[batch_size:]), 1)
        layers_rgba         = torch.stack(layers_rgba, 1)
        layers_rgba         = torch.stack((layers_rgba[:batch_size], layers_rgba[batch_size:]), 1)
        layers_flow         = torch.stack(layers_flow, 1)
        layers_flow         = torch.stack((layers_flow[:batch_size], layers_flow[batch_size:]), 1)
        layers_alpha_warped = torch.stack(layers_alpha_warped, 1)
        brightness_scale    = torch.stack((brightness_scale[:batch_size], brightness_scale[batch_size:]), 1)
        background_offset   = torch.stack((background_offset[:batch_size], background_offset[batch_size:]), 1)


        out = {
            "rgba_reconstruction": composite_rgba,          # [B, 2, 4, H, W]
            "flow_reconstruction": composite_flow,          # [B, 2, 2, H, w]
            "reconstruction_warped": composite_warped,      # [B, 3, H, W]
            "layers_rgba": layers_rgba,                     # [B, 2, L, 4, H, W]
            "layers_flow": layers_flow,                     # [B, 2, L, 2, H, W]
            "layers_alpha_warped": layers_alpha_warped,     # [B, L, 1, H, W]
            "brightness_scale": brightness_scale,           # [B, 2, 1, H, W]
            "background_offset": background_offset          # [B, 2, 2, H, W]
        }
        return out

    def get_alpha_from_rgba(self, rgba):
        """
        Get the alpha layer from an rgba tensor that is ready for compositing
        """
        return rgba[:, 3:4] * .5 + .5

    def get_background_offset(self, jitter_grid, index):
        background_offset = F.interpolate(self.bg_offset, (self.max_frames, 4, 7), mode="trilinear")
        background_offset = background_offset[0, :, index].transpose(0, 1)
        background_offset = F.grid_sample(background_offset, jitter_grid.permute(0, 2, 3, 1), align_corners=True)

        return background_offset

    def get_brightness_scale(self, jitter_grid, index):
        brightness_scale = F.interpolate(self.brightness_scale, (self.max_frames, 4, 7), mode='trilinear', align_corners=True)
        brightness_scale = brightness_scale[0, 0, index].unsqueeze(1)
        brightness_scale = F.grid_sample(brightness_scale, jitter_grid.permute(0, 2, 3, 1), align_corners=True)

        return brightness_scale