from os import path
import torch
import torch.nn.functional as F
from math import ceil

class BackgroundVolume(object):

    def __init__(self,
                 save_dir: str,
                 num_frames: int,
                 in_channels: int = 16,
                 num_static_channels: int = 5,
                 temporal_coarseness: int = 2,
                 upsample_size: int = 16,
                 frame_size: list = [864, 480],
                 use_depth: bool = False) -> None:
        super().__init__()
        assert num_frames > temporal_coarseness, "Number of frames in the video must be greater than the temporal coarseness"

        self.save_dir  = save_dir

        if path.exists(path.join(save_dir, "spatiotemporal_noise.pth")) and 1 == 0:
            self.spatial_noise           = torch.load(path.join(save_dir, "spatial_noise.pth"))
            self.spatial_noise_upsampled = torch.load(path.join(save_dir, "spatial_noise_upsampled.pth"))
            self.spatiotemporal_noise    = torch.load(path.join(save_dir, "spatiotemporal_noise.pth"))
        else:
            self.spatial_noise = torch.randn(in_channels - 3 - int(use_depth), frame_size[1] // upsample_size, frame_size[0] // upsample_size)
            self.spatial_noise_upsampled = F.interpolate(self.spatial_noise.unsqueeze(0), (frame_size[1], frame_size[0]), mode='bilinear')[0]

            if num_static_channels < in_channels: # always true except with Omnimatte
                spatiotemporal_noise = torch.randn(in_channels - num_static_channels - 3 - int(use_depth), ceil(num_frames/temporal_coarseness), frame_size[1] // upsample_size, frame_size[0] // upsample_size)
                spatiotemporal_noise_upsampled = F.interpolate(spatiotemporal_noise.unsqueeze(0), (num_frames, frame_size[1], frame_size[0]), mode='trilinear')[0]
                self.spatiotemporal_noise = torch.cat((self.spatial_noise_upsampled[:num_static_channels].unsqueeze(1).repeat(1, num_frames, 1, 1), spatiotemporal_noise_upsampled))
            else:
                self.spatiotemporal_noise = self.spatial_noise_upsampled.unsqueeze(1).repeat(1, num_frames, 1, 1)

            torch.save(self.spatial_noise,           path.join(save_dir, "spatial_noise.pth"))
            torch.save(self.spatial_noise_upsampled, path.join(save_dir, "spatial_noise_upsampled.pth"))
            torch.save(self.spatiotemporal_noise,    path.join(save_dir, "spatiotemporal_noise.pth"))
