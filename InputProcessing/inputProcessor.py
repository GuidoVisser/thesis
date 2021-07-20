import torch
import cv2
import numpy as np
from os import path, listdir

from utils.video_utils import opencv_folder_to_video
from .BackgroundAttentionVolume.backgroundVolume import BackgroundVolume
from .MaskPropagation.maskHandler import MaskHandler
from .FlowHandler.flowHandler import FlowHandler
from .frameIterator import FrameIterator

class InputProcessor(object):
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 initial_mask: str,
                 flow_dir: str,
                 background_dir: str,
                 frame_size: list=[864, 480]
                ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.flow_dir = flow_dir
        self.background_dir = background_dir
        self.frame_iterator = FrameIterator(img_dir, frame_size)

        self.mask_handler = MaskHandler(img_dir, mask_dir, initial_mask, frame_size)
        self.background_volume = BackgroundVolume(img_dir, mask_dir, "cuda:0", save_dir=background_dir)
        self.flow_handler = FlowHandler(self.frame_iterator, self.background_volume.homographies, flow_dir)

    def get_frame_input(self, frame_idx):
        """
        Return the input for a single frame
        """
        img = self.frame_iterator[frame_idx]
        mask = self.mask_handler.get_binary_mask(frame_idx)

        matte = self.get_rgba_matte(img, mask)

        noise = self.background_volume.get_frame_noise(frame_idx).astype(np.uint8)

        flow = self.flow_handler.calculate_flow_between_stabalized_frames(frame_idx)
        flow_img = self.flow_handler.convert_flow_to_image(flow, bgr=True)
        flow_matte = self.get_flow_matte(flow, mask)
        flow_matte_img = self.flow_handler.convert_flow_to_image(flow_matte, bgr=True)


        return img, flow_img, mask, matte, flow_matte_img, noise
        

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

    def get_rgba_matte(self, img, a_mask):
        """
        Get an rgba matte from an image and an alpha-mask
        """
        return img * np.stack([a_mask]*3, 2).astype(np.uint8)


    def get_flow_matte(self, flow, a_mask):
        """
        Get a matte for flow using an alpha mask and the flow
        """
        return flow * np.stack([a_mask]*2, 2).astype(np.float32)