from os import path, listdir
import cv2
import torch
import numpy as np

class FrameIterator(object):

    def __init__(self, 
                 img_dir:str, 
                 frame_size: list=[864, 480]
                ) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.images = [path.join(img_dir, frame) for frame in sorted(listdir(img_dir))]

        self.frame_size = frame_size

    def __getitem__(self, idx):
        img = cv2.resize(cv2.imread(self.images[idx]), self.frame_size)
        return img
    
    def __len__(self):
        return len(self.images)
