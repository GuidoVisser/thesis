import torch
import torch.nn.functional as F
import cv2
import numpy as np
from math import ceil
from os import path

from InputProcessing.homography import HomographyHandler
from utils.utils import create_dir

class BackgroundVolume(object):

    def __init__(self,
                 args,
                 homography_handler: HomographyHandler) -> None:
        super().__init__()

        num_frames = len(homography_handler)
        assert num_frames > args.noise_temporal_coarseness, "Number of frames in the video must be greater than the temporal coarseness"

        self.save_dir = path.join(args.out_dir, "background")
        create_dir(self.save_dir)

        if path.exists(path.join(self.save_dir, "spatiotemporal_noise.pth")):
            self.spatial_noise           = torch.load(path.join(self.save_dir, "spatial_noise.pth"))
            self.spatial_noise_upsampled = torch.load(path.join(self.save_dir, "spatial_noise_upsampled.pth"))
            self.spatiotemporal_noise    = torch.load(path.join(self.save_dir, "spatiotemporal_noise.pth"))
        else:
            self.spatial_noise = torch.randn(args.in_channels - 3 - int(args.use_depth), args.frame_height // args.noise_upsample_size, args.frame_width // args.noise_upsample_size)
            self.spatial_noise_upsampled = F.interpolate(self.spatial_noise.unsqueeze(0), (args.frame_height, args.frame_width), mode='bilinear')[0]

            if args.num_static_channels < args.in_channels - 3 - int(args.use_depth): # always true except with Omnimatte
                spatiotemporal_noise = torch.randn(args.in_channels - args.num_static_channels - 3 - int(args.use_depth), ceil(num_frames/args.noise_temporal_coarseness), args.frame_height // args.noise_upsample_size, args.frame_width // args.noise_upsample_size)
                spatiotemporal_noise_upsampled = F.interpolate(spatiotemporal_noise.unsqueeze(0), (num_frames, args.frame_height, args.frame_width), mode='trilinear')[0]
                self.spatiotemporal_noise = torch.cat((self.spatial_noise_upsampled[:args.num_static_channels].unsqueeze(1).repeat(1, num_frames, 1, 1), spatiotemporal_noise_upsampled))
            else:
                self.spatiotemporal_noise = self.spatial_noise_upsampled.unsqueeze(1).repeat(1, num_frames, 1, 1)
            
            torch.save(self.spatial_noise,           path.join(self.save_dir, "spatial_noise.pth"))
            torch.save(self.spatial_noise_upsampled, path.join(self.save_dir, "spatial_noise_upsampled.pth"))
            torch.save(self.spatiotemporal_noise,    path.join(self.save_dir, "spatiotemporal_noise.pth"))

        self.spatiotemporal_noise_uv_sampled = F.grid_sample(self.spatiotemporal_noise.permute(1, 0, 2, 3), homography_handler.uv_maps).permute(1, 0, 2, 3)

        self.homography_handler = homography_handler

    def visualize(self, frame_idx):
        spatial_noise_img = (self.spatial_noise_upsampled[:3].permute(1, 2, 0).numpy() * .5 + .5) * 255
        spatiotemporal_noise_img = (self.spatiotemporal_noise[:3, frame_idx].permute(1, 2, 0).numpy() * .5 + .5) * 255
        sampled_noise = (self.spatiotemporal_noise_uv_sampled[:3, frame_idx].permute(1, 2, 0).numpy() * .5 + .5) * 255

        uv = self.homography_handler.get_frame_uv(frame_idx).clone().cpu().numpy()
        corners = [uv[0, 0], uv[0, -1], uv[-1, -1], uv[-1, 0]]

        corners = np.array([(c * .5 + .5) * np.array([448, 256]) for c in corners]).reshape((-1, 1, 2)).astype(np.int32)

        color = (255, 255, 255)
        thickness = 3

        spatiotemporal_noise_img = cv2.polylines(np.ascontiguousarray(spatiotemporal_noise_img), [corners], True, color, thickness, cv2.LINE_AA)

        cv2.imwrite(f"{self.save_dir}/spatial_noise.png", spatial_noise_img)
        cv2.imwrite(f"{self.save_dir}/{frame_idx:05}.png", spatiotemporal_noise_img)
        cv2.imwrite(f"{self.save_dir}/sampled_{frame_idx:05}.png", sampled_noise)
