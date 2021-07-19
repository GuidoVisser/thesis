import torch
import cv2
import numpy as np
from os import path, listdir

from .BackgroundAttentionVolume.backgroundVolume import BackgroundVolume
from .MaskPropagation.maskHandler import MaskHandler
from .FlowHandler.flowHandler import FlowHandler
from .frameIterator import FrameIterator

class InputProcessor(object):
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
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

        self.background_volume = BackgroundVolume(img_dir, mask_dir, "cuda:0", save_dir=background_dir)
        self.mask_handler = MaskHandler(img_dir, mask_dir, frame_size)
        self.flow_handler = FlowHandler(self.frame_iterator, flow_dir)

    def get_frame_input(self, frame_idx):
        """
        Return the input for a single frame
        """
        img = self.frame_iterator[frame_idx]
        mask = self.mask_handler.get_binary_mask(frame_idx)

        matte = img * np.stack([mask]*3, 2).astype(np.uint8)

        noise = self.background_volume.get_frame_noise(frame_idx).astype(np.uint8)

        self.flow_handler.calculate_flow_for_video()
        flow = self.flow_handler.load_flow_image(frame_idx)
        flow_matte = flow * np.stack([mask]*3, 2).astype(np.uint8)

        cv2.imshow("test", img)
        cv2.imshow("mask", mask)
        cv2.imshow("matte", matte)
        cv2.imshow("noise", noise)
        cv2.imshow("flow", flow)
        cv2.imshow("flow matte", flow_matte)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

        