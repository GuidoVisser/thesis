from os import path, listdir
import torch
import numpy as np
import cv2
from typing import Union
from InputProcessing.maskHandler import MaskHandler

from utils.utils import create_dir
from models.third_party.MonoDepth.models_resnet import Resnet18_md
from models.third_party.MonoDepth.utils import prepare_dataloader


class DepthHandler(object):
    """
    Dataset object that prepares the depth estimation of the video and can be queried for the depth of given frames
    """

    def __init__(self, 
                 img_dir: str, 
                 out_dir: str, 
                 args,
                 mask_handler: MaskHandler) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.out_dir = out_dir
        
        # if not path.exists(self.out_dir):
        create_dir(out_dir)

        # Set up model for depth estimation
        self.device = args.device

        self.model_weights = args.depth_model

        # Load data
        self.frame_size = (args.frame_width, args.frame_height)

        self.n_img, self.loader = prepare_dataloader(img_dir, (args.frame_height, args.frame_width))
        self.estimate_depth()

        self.frame_paths = [path.join(out_dir, fn) for fn in sorted(listdir(self.out_dir))]

        self.mask_handler = mask_handler

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:

        if isinstance(idx, slice):
            depth = []
            for frame_path in self.frame_paths[idx]:
                depth.append(torch.from_numpy(cv2.resize(np.float32(cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)), self.frame_size)) / 255.)
            depth = torch.stack(depth)

        else:
            depth = torch.from_numpy(cv2.resize(np.float32(cv2.imread(self.frame_paths[idx], cv2.IMREAD_GRAYSCALE)), self.frame_size)) / 255.

        depth = 1 - depth.unsqueeze(0)

        _, masks = self.mask_handler[idx]
        if len(depth.shape) == 4:
            object_depth = masks * depth.repeat(masks.shape[0], 1, 1, 1, 1)
        elif len(depth.shape) == 3:
            object_depth = masks * depth.repeat(masks.shape[0], 1, 1, 1)
        else:
            raise ValueError(f"Incorrect shape of depth estimation: {depth.shape}")

        depth        = depth * 2 - 1
        object_depth = object_depth * 2 - 1

        return depth, object_depth

    def __len__(self) -> None:
        return len(listdir(self.img_dir))


    @ torch.no_grad()
    def estimate_depth(self):

        model = Resnet18_md(num_in_layers=3).to(self.device).eval()
        if torch.cuda.device_count() == 0:
            model.load_state_dict(torch.load(self.model_weights, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(self.model_weights))
        
        disparities = np.zeros((self.n_img,
                                  self.frame_size[1], self.frame_size[0]),
                                  dtype=np.float32)
        for i, data in enumerate(self.loader):
            
            # Get the inputs
            data = data.to(self.device).squeeze()
            
            # Do a forward pass
            disps = model(data)
            disparities[i] = self.post_process_disparity(disps[0][:, 0, :, :].cpu().numpy())
            
        disparities /= np.max(disparities)
        
        for i in range(len(disparities)):
            disparity_img = cv2.resize(disparities[i], self.frame_size)
            cv2.imwrite(path.join(self.out_dir, f"{i:05}.png"), disparity_img * 255)

    def post_process_disparity(self, disp):
        
        # get dimensions
        _, h, w = disp.shape

        # separate lefthand disparity from righthand disparity and get the mean disparity
        l_disp = disp[0]
        r_disp = np.fliplr(disp[1])
        m_disp = 0.5 * (l_disp + r_disp)
        
        # initialize grid
        grid, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        
        # get masks
        l_mask = 1.0 - np.clip(20 * (grid - 0.05), 0, 1)
        r_mask = np.fliplr(l_mask)
        
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp    
