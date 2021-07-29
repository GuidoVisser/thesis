import torch
import cv2
import numpy as np
from os import path
from typing import Union

from utils.video_utils import opencv_folder_to_video
from .BackgroundAttentionVolume.backgroundVolume import BackgroundVolume
from .MaskPropagation.maskHandler import MaskHandler
from .FlowHandler.flowHandler import FlowHandler
from .frameIterator import FrameIterator

class InputProcessor(object):
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 initial_mask: Union[str, list],
                 flow_dir: str,
                 background_dir: str,
                 device: str = "cuda",
                 frame_size: list=[864, 480]
                ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.flow_dir = flow_dir
        self.background_dir = background_dir
        self.device = device
        self.frame_iterator = FrameIterator(img_dir, frame_size, device=self.device)

        if isinstance(initial_mask, str):
            self.N_objects = 1
        else:
            self.N_objects = len(initial_mask)

        self.mask_handler = MaskHandler(img_dir, mask_dir, initial_mask, frame_size, device=self.device)
        self.background_volume = BackgroundVolume(img_dir, mask_dir, self.device, save_dir=background_dir)
        self.flow_handler = FlowHandler(self.frame_iterator, self.mask_handler, self.background_volume.homographies, flow_dir, device=self.device)

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
        # Get RGB input
        rgb_t0 = self.frame_iterator[idx]
        rgb_t1 = self.frame_iterator[idx + 1]
        rgb = torch.stack((rgb_t0, rgb_t1)).to(self.device)

        # Get mask input
        masks = self.mask_handler[idx]
        
        # Get flow input and confidence
        flow, flow_conf = self.flow_handler[idx]

        # Get noise input
        noise_t0 = torch.from_numpy(np.float64(self.background_volume.get_frame_noise(idx))).permute(2, 0, 1) / 255.
        noise_t1 = torch.from_numpy(np.float64(self.background_volume.get_frame_noise(idx + 1))).permute(2, 0, 1) / 255.
        noise = torch.stack((noise_t0, noise_t1)).to(self.device).unsqueeze(1).repeat(1, self.N_objects, 1, 1, 1)

        # Get object layers for RGB and flow
        flow_layers = self.get_flow_layers(flow, masks)
        rgb_layers = self.get_rgb_layers(rgb, masks)

        model_input = torch.cat((rgb_layers, flow_layers, noise), 2)

        targets = {
            "rgb": rgb,
            "flow": flow,
            "masks": masks,
            "flow_confidence": flow_conf
        }

        return model_input, targets

    def __len__(self):
        return len(self.frame_iterator)

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
        