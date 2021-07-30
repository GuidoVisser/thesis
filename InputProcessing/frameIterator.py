from os import path, listdir
import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F

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
        img = torch.from_numpy(cv2.resize(np.float32(cv2.imread(self.images[idx])), self.frame_size)).permute(2, 0, 1) / 255.
        img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return img.to(self.device)
    
    def __len__(self):
        return len(self.images) - 1 

    def get_np_frame(self, idx):
        img = cv2.resize(cv2.imread(self.images[idx]), self.frame_size)
        return img