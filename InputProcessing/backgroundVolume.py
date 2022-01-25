from os import path
import torch
import torch.nn.functional as F
from math import ceil

class BackgroundVolume(object):

    def __init__(self,
                 save_dir: str,
                 num_frames: int,
                 args) -> None:
        super().__init__()
        assert num_frames > args.noise_temporal_coarseness, "Number of frames in the video must be greater than the temporal coarseness"

        self.save_dir  = save_dir

        if path.exists(path.join(save_dir, "spatiotemporal_noise.pth")):
            self.spatial_noise           = torch.load(path.join(save_dir, "spatial_noise.pth"))
            self.spatial_noise_upsampled = torch.load(path.join(save_dir, "spatial_noise_upsampled.pth"))
            self.spatiotemporal_noise    = torch.load(path.join(save_dir, "spatiotemporal_noise.pth"))
        else:
            self.spatial_noise = torch.randn(args.in_channels - 3 - int(args.use_depth), args.frame_height // args.noise_upsample_size, args.frame_width // args.noise_upsample_size)
            self.spatial_noise_upsampled = F.interpolate(self.spatial_noise.unsqueeze(0), (args.frame_height, args.frame_width), mode='bilinear')[0]

            if args.num_static_channels < args.in_channels - 3 - int(args.use_depth): # always true except with Omnimatte
                spatiotemporal_noise = torch.randn(args.in_channels - args.num_static_channels - 3 - int(args.use_depth), ceil(num_frames/args.noise_temporal_coarseness), args.frame_height // args.noise_upsample_size, args.frame_width // args.noise_upsample_size)
                spatiotemporal_noise_upsampled = F.interpolate(spatiotemporal_noise.unsqueeze(0), (num_frames, args.frame_height, args.frame_width), mode='trilinear')[0]
                self.spatiotemporal_noise = torch.cat((self.spatial_noise_upsampled[:args.num_static_channels].unsqueeze(1).repeat(1, num_frames, 1, 1), spatiotemporal_noise_upsampled))
            else:
                self.spatiotemporal_noise = self.spatial_noise_upsampled.unsqueeze(1).repeat(1, num_frames, 1, 1)

            torch.save(self.spatial_noise,           path.join(save_dir, "spatial_noise.pth"))
            torch.save(self.spatial_noise_upsampled, path.join(save_dir, "spatial_noise_upsampled.pth"))
            torch.save(self.spatiotemporal_noise,    path.join(save_dir, "spatiotemporal_noise.pth"))
