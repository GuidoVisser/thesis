import torch
import torch.nn as nn 
import numpy as np
import cv2

from typing import Union
from os import path, listdir


class ImprovementLoader(object):
    """
    Dataset object for the photometric improvement GAN
    """

    def __init__(self, 
                 root: str,
                 separate_bg: bool,
                 extention: str = "png") -> None:
        super().__init__()

        self.root = root
        self.separate_bg = separate_bg
        self.ext = extention

    def __getitem__(self, idx: int) -> dict:
        """
        Get one instance of the model input for the Photometric Improvement GAN
        
        Args:
            idx (int)
        """

        ground_truth = self.get_frame_from_dir("ground_truth", idx)

        rgba = self.get_frame_from_dir("background", idx).unsqueeze(0)
        rgba = torch.cat((rgba, torch.ones_like(rgba[:, 0:1])), dim=1)
        for object_dir in sorted(listdir(path.join(self.root, "foreground"))):
            if object_dir == "01" and self.separate_bg:
                continue

            new_layer = self.get_frame_from_dir(f"foreground/{object_dir}", idx).unsqueeze(0)

            rgba = torch.cat((rgba, new_layer))
        
        return rgba, ground_truth, idx

    def __len__(self) -> None:
        return len(listdir(path.join(self.root, "background")))

    def get_frame_from_dir(self, dir: str, idx: int):
        """
        Load and process the i-th frame from the given directory

        Args:
            dir (str): target directory relative to self.root
            idx (int): index of frame
        """
        return torch.from_numpy(cv2.imread(path.join(self.root, dir, f"{idx:05}.{self.ext}"), cv2.IMREAD_UNCHANGED) / 255 * 2 - 1).float().permute(2, 0, 1)
