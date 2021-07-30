from posix import listdir
import torch
import cv2
import numpy as np
from os import path
from typing import Union

from utils.video_utils import opencv_folder_to_video
from utils.utils import create_dirs
from .BackgroundAttentionVolume.backgroundVolume import BackgroundVolume
from .MaskPropagation.maskHandler import MaskHandler
from .FlowHandler.flowHandler import FlowHandler
from .frameIterator import FrameIterator

class InputProcessor(object):
    def __init__(self,
                 video: str,
                 out_root: str,
                 initial_mask: Union[str, list],
                 in_channels: int = 16,
                 device: str = "cuda",
                 frame_size: list=[448, 256]
                ) -> None:
        super().__init__()

        self.device      = device
        self.frame_size  = frame_size
        self.in_channels = in_channels

        if isinstance(initial_mask, str):
            self.N_objects = 1
        else:
            self.N_objects = len(initial_mask)

        img_dir        = path.join(out_root, "images")
        mask_dir       = path.join(out_root, "masks")
        flow_dir       = path.join(out_root, "flow")
        background_dir = path.join(out_root, "background")
        create_dirs(img_dir, mask_dir, flow_dir, background_dir)

        self.prepare_image_dir(video, img_dir)

        self.frame_iterator    = FrameIterator(img_dir, frame_size, device=self.device)
        self.mask_handler      = MaskHandler(video, mask_dir, initial_mask, frame_size, device=self.device)
        self.background_volume = BackgroundVolume(img_dir, mask_dir, background_dir, self.device, in_channels=in_channels, frame_size=self.frame_size)
        self.flow_handler      = FlowHandler(self.frame_iterator, self.mask_handler, self.background_volume.homographies, flow_dir, device=self.device)

        self.composite_order = [tuple(range(1, self.N_objects + 1))] * len(self.frame_iterator)


    def get_frame_input(self, frame_idx):
        """
        Return the input for a single frame
        """
        img = self.frame_iterator[frame_idx]
        mask = self.mask_handler.get_binary_mask(frame_idx)

        matte = self.get_rgba_matte(img, mask)

        noise = self.background_volume.get_frame_noise(frame_idx).astype(np.uint8)

        flow, conf = self.flow_handler.calculate_flow_between_stabilized_frames(frame_idx)
        flow_img = self.flow_handler.convert_flow_to_image(flow, bgr=True)
        flow_matte = self.get_flow_matte(flow, mask)
        flow_matte_img = self.flow_handler.convert_flow_to_image(flow_matte, bgr=True)

        return img, flow_img, mask, matte, flow_matte_img, conf, noise
        
    def __getitem__(self, idx):
        """
        Get the optical flow input for the layer decompostion model

        Dimensions:
            T: time, the model uses frame at t and t+1 as input
            L: number of layers, always equal to N_objects + 1 (background)
            C: channel dimension, amount of channels in the input
            H: height of the image
            W: width of the image

        """
        # Get RGB input
        rgb_t0 = self.frame_iterator[idx]
        rgb_t1 = self.frame_iterator[idx + 1]
        rgb    = torch.stack((rgb_t0, rgb_t1)) # [T, C, H, W] = [2, 3, H, W]

        # Get mask input
        masks_t0 = self.mask_handler[idx]
        masks_t1 = self.mask_handler[idx + 1]
        masks    = torch.stack((masks_t0, masks_t1)) # [T, L-1, C, H, W] = [2, L-1, 1, H, W]
        
        # Get flow input and confidence
        flow_t0, flow_conf_t0, object_flow_t0, background_flow_t0 = self.flow_handler[idx]
        flow_t1, flow_conf_t1, object_flow_t1, background_flow_t1 = self.flow_handler[idx + 1]
        
        flow            = torch.stack((flow_t0, flow_t1))                       # [T, C, H, W]      = [2, 2, H, W]
        flow_conf       = torch.stack((flow_conf_t0, flow_conf_t1))             # [T, L-1, H, W]    = [2, L-1, H, W]
        object_flow     = torch.stack((object_flow_t0, object_flow_t1))         # [T, L-1, C, H, W] = [2, L-1, 2, H, W]
        background_flow = torch.stack((background_flow_t0, background_flow_t1)) # [T, C, H, W]      = [2, 2, H, W]
        
        # Get noise input
        background_noise = torch.from_numpy(self.background_volume.spatial_noise_upsampled).float().permute(2, 0, 1).to(self.device)
        background_noise = background_noise.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1, 1)

        noise_t0 = torch.from_numpy(np.float32(self.background_volume.get_frame_noise(idx))).permute(2, 0, 1)       
        noise_t1 = torch.from_numpy(np.float32(self.background_volume.get_frame_noise(idx + 1))).permute(2, 0, 1)
        noise = torch.stack((noise_t0, noise_t1)).to(self.device).unsqueeze(1).repeat(1, self.N_objects, 1, 1, 1)
        
        background_uv_map_t0 = self.background_volume.get_frame_uv(idx)
        background_uv_map_t1 = self.background_volume.get_frame_uv(idx + 1)
        background_uv_map = torch.stack((background_uv_map_t0, background_uv_map_t1))

        # Get model input
        pids = torch.Tensor(self.composite_order[idx]).view(1, -1, 1, 1, 1).to(self.device) * masks  # [T, L-1, 1, H, W] = [2, L-1, 1, H, W]
        input_tensor = torch.cat((pids, object_flow, noise), dim=2)                                  # [T, L-1, C, H, W] = [2, L-1, 16, H, W]
        background_input = torch.cat((torch.zeros((2, 1, 3, self.frame_size[1], self.frame_size[0]), device=self.device, dtype=torch.float32), background_noise), dim=2)
        input_tensor = torch.cat((background_input, input_tensor), dim=1)

        targets = {
            "rgb": rgb,
            "flow": flow,
            "masks": masks,
            "flow_confidence": flow_conf
        }

        model_input = {
            "input_tensor": input_tensor,
            "background_flow": background_flow,
            "background_uv_map": background_uv_map
        }

        return model_input, targets

    def __len__(self):
        return len(self.frame_iterator) - 1

    def input_demo(self, directory):
        """
        Generate a demo video of the generated input and ground truth values
        """

        for i in range(len(self.frame_iterator) - 1):
            print(f"{i} / {len(self.frame_iterator) - 1}")
            img, flow, mask, matte, flow_matte, noise = self.get_frame_input(i)

            gt = np.concatenate((img, flow))
            mid = np.concatenate((matte, flow_matte))
            right = np.concatenate((np.stack([mask]*3, 2).astype(np.uint8)*255, noise))

            full = np.concatenate((gt, mid, right), axis=1)

            cv2.imwrite(path.join(directory, f"{i:05}.png"), full)
        
        opencv_folder_to_video(directory, path.join(directory, "demo.mp4"))

    def get_rgb_layers(self, rgb, masks):
        """
        Get an rgba matte for every object
        
        Args:
            rgb (torch.Tensor[2, 3, H, W])
            a_mask (torch.Tensor[2, L, H, W])

        Returns:
            rgb_matte (torch.Tensor[2, L, 3, H, W])
        """
        masks = torch.stack([masks]*3, 2)
        rgb = torch.stack([rgb]*self.N_objects, 1)

        return rgb * masks  


    def get_flow_layers(self, flow: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Get a matte for the optical flow of every object

        Args:
            flow (torch.Tensor[2, 2, H, W])
            a_mask (torch.Tensor[2, L, H, W])

        Returns:
            flow_matte (torch.Tensor[2, L, 2, H, W])
        """
        masks = torch.stack([masks]*2, 2)
        flow = torch.stack([flow]*self.N_objects, 1)

        return flow * masks        
        
    def prepare_image_dir(self, video_dir, out_dir):

        frame_paths = [frame for frame in sorted(listdir(video_dir))]
        
        for frame_path in frame_paths:
            img = cv2.resize(cv2.imread(path.join(video_dir, frame_path)), self.frame_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(path.join(out_dir, frame_path), img)