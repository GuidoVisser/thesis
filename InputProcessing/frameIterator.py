from os import path, listdir
import cv2
import torch
import numpy as np

class FrameIterator(object):

    def __init__(self, 
                 img_dir:str, 
                 frame_size: list=[864, 480],
                 device: str = "cuda"
                ) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.images = [path.join(img_dir, frame) for frame in sorted(listdir(img_dir))]

        self.frame_size = frame_size
        self.device = device

    def __getitem__(self, idx):
        img = torch.from_numpy(cv2.resize(np.float32(cv2.cvtColor(cv2.imread(self.images[idx]), cv2.COLOR_BGR2RGB)), self.frame_size)).permute(2, 0, 1) / 255.
        img = img * 2 - 1
        
        return img
    
    def __len__(self):
        return len(self.images)

    def get_np_frame(self, idx):
        img = cv2.resize(cv2.imread(self.images[idx]), self.frame_size)
        return img