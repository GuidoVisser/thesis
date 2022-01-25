from os import path, listdir
from argparse import Namespace
from typing import Union
import cv2
import torch
import numpy as np

class FrameIterator(object):

    def __init__(self, 
                 img_dir:str, 
                 args: Namespace,
                ) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.images = [path.join(img_dir, frame) for frame in sorted(listdir(img_dir))]

        self.frame_size = (args.frame_width, args.frame_height)
        self.device = args.device

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        
        if isinstance(idx, slice):
            img_paths = self.images[idx]
            img = []
            for img_path in img_paths:
                img.append(torch.from_numpy(cv2.resize(np.float32(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)), self.frame_size)).permute(2, 0, 1) / 255.)
            img = torch.stack(img, dim=1)
            img = img * 2 - 1
        else:
            img = torch.from_numpy(cv2.resize(np.float32(cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)), self.frame_size)).permute(2, 0, 1) / 255.
            img = img * 2 - 1
        
        return img
    
    def __len__(self):
        return len(self.images)

    def get_np_frame(self, idx):
        img = cv2.resize(cv2.imread(self.images[idx]), self.frame_size)
        return img