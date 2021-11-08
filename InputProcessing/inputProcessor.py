from posix import listdir
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from os import path
from typing import Union

from utils.video_utils import opencv_folder_to_video
from utils.utils import create_dirs
from .backgroundVolume import BackgroundVolume
from .maskHandler import MaskHandler
from .flowHandler import FlowHandler
from .frameIterator import FrameIterator
from .homography import HomographyHandler

class InputProcessor(object):
    def __init__(self,
                 video: str,
                 out_root: str,
                 initial_mask: Union[str, list],
                 composite_order_fp: Union[str, None],
                 propagation_model: str,
                 flow_model: str,
                 in_channels: int = 16,
                 noise_temporal_coarseness: int = 2,
                 device: str = "cuda",
                 frame_size: tuple=(448, 256),
                 do_jitter: bool = True,
                 jitter_rate: float = 0.75
                ) -> None:
        super().__init__()

        self.device      = device
        self.frame_size  = frame_size
        self.in_channels = in_channels
        self.do_jitter   = do_jitter
        self.jitter_rate = jitter_rate
        self.jitter_mode = 'bilinear'

        if isinstance(initial_mask, str):
            self.N_objects = 1
        else:
            self.N_objects = len(initial_mask)

        self.N_layers = self.N_objects + 2

        self.img_dir   = path.join(out_root, "images")
        mask_dir       = path.join(out_root, "masks")
        flow_dir       = path.join(out_root, "flow")
        background_dir = path.join(out_root, "background")
        create_dirs(self.img_dir, mask_dir, flow_dir, background_dir)

        self.prepare_image_dir(video, self.img_dir)

        self.frame_iterator     = FrameIterator(self.img_dir, frame_size, device=self.device)
        self.mask_handler       = MaskHandler(video, mask_dir, initial_mask, frame_size, device=self.device, propagation_model=propagation_model)
        self.flow_handler       = FlowHandler(self.frame_iterator, self.mask_handler, flow_dir, raft_weights=flow_model, device=self.device)
        self.homography_handler = HomographyHandler(out_root, self.img_dir, path.join(flow_dir, "dynamics_mask"), self.device, frame_size)
        self.background_volume  = BackgroundVolume(background_dir, num_frames=len(self.frame_iterator), in_channels=in_channels, temporal_coarseness=noise_temporal_coarseness, frame_size=frame_size)       

        self.load_composite_order(composite_order_fp)
        
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
        masks_t0, binary_masks_t0 = self.mask_handler[idx]
        masks_t1, binary_masks_t1 = self.mask_handler[idx + 1]
        masks                     = torch.stack((masks_t0, masks_t1))               # [T, L-2, C, H, W] = [2, L-2, 1, H, W]
        binary_masks              = torch.stack((binary_masks_t0, binary_masks_t1)) # [T, L-2, C, H, W] = [2, L-2, 1, H, W]

        # Get flow input and confidence
        flow_t0, flow_conf_t0, object_flow_t0, dynamics_t0 = self.flow_handler[idx]
        flow_t1, flow_conf_t1, object_flow_t1, dynamics_t1 = self.flow_handler[idx + 1]

        background_flow_t0 = self.homography_handler.frame_homography_to_flow(idx)
        background_flow_t1 = self.homography_handler.frame_homography_to_flow(idx +1)
        
        flow            = torch.stack((flow_t0, flow_t1))                       # [T,      C, H, W] 
        flow_conf       = torch.stack((flow_conf_t0, flow_conf_t1))             # [T, L-2,    H, W] 
        object_flow     = torch.stack((object_flow_t0, object_flow_t1))         # [T, L-2, C, H, W] 
        background_flow = torch.stack((background_flow_t0, background_flow_t1)) # [T,      C, H, W] 
        # dynamics_mask   = torch.stack((dynamics_t0, dynamics_t1))               # [T,         H, W]    
        
        # convert dynamics mask to trimap
        # dynamics_mask = dynamics_mask * 2 - 1

        # Add background layers to masks
        masks = torch.cat((torch.zeros_like(masks[:, 0:1]).repeat(1, 2, 1, 1, 1), masks), dim=1)

        # Get spatial noise input
        background_noise = self.background_volume.spatial_noise_upsampled                   #       [C-3, H, W]
        background_noise = background_noise.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1, 1) # [T, 1, C-3, H, W]

        background_uv_map_t0 = self.homography_handler.get_frame_uv(idx)
        background_uv_map_t1 = self.homography_handler.get_frame_uv(idx + 1)
        background_uv_map    = torch.stack((background_uv_map_t0, background_uv_map_t1))

        # Get spatiotemporal noise input
        spatiotemporal_noise    = self.background_volume.spatiotemporal_noise #    [Nf, C-3, H, W]
        spatiotemporal_noise_t0 = F.grid_sample(spatiotemporal_noise[idx:idx+1],   background_uv_map_t0.unsqueeze(0)) #    [1, C-3, H, w]
        spatiotemporal_noise_t1 = F.grid_sample(spatiotemporal_noise[idx+1:idx+2], background_uv_map_t0.unsqueeze(0)) #    [1, C-3, H, w]
        spatiotemporal_noise_t  = torch.stack((spatiotemporal_noise_t0, spatiotemporal_noise_t1))                     # [T, 1, C-3, H, W]

        # Get model input
        zeros = torch.zeros((2, 1, 3, self.frame_size[1], self.frame_size[0]), dtype=torch.float32)  # [T, 1, 3, H, W]
        static_background_input  = torch.cat((zeros, background_noise), dim=2)                       # [T, 1, C, H, W]
        dynamic_background_input = torch.cat((zeros, spatiotemporal_noise_t), dim=2)                 # [T, 1, C, H, W]
        
        pids = binary_masks * (torch.Tensor(self.composite_order[idx]) + 1).view(1, self.N_layers - 2, 1, 1, 1)             # [T, L-2, 1, H, W] 
        input_tensor = torch.cat((pids, object_flow, spatiotemporal_noise_t.repeat(1, self.N_layers - 2, 1, 1, 1)), dim=2)  # [T, L-2, C, H, W]
        
        input_tensor = torch.cat((static_background_input, dynamic_background_input, input_tensor), dim=1)                  # [T, L, C, H, W]

        spatiotemporal_noise = torch.cat((torch.zeros_like(spatiotemporal_noise[:, :3]), spatiotemporal_noise), dim=1) # [Nf, C, H, W]

        # get parameters for jitter
        params = self.get_jitter_parameters()
            
        # transform targets
        rgb       = self.apply_jitter_transform(rgb,       params)
        flow      = self.apply_jitter_transform(flow,      params)
        masks     = self.apply_jitter_transform(masks,     params)
        flow_conf = self.apply_jitter_transform(flow_conf, params)

        # transform inputs
        input_tensor      = self.apply_jitter_transform(input_tensor,    params)
        background_flow   = self.apply_jitter_transform(background_flow, params)
        background_uv_map = self.apply_jitter_transform(background_uv_map.permute(0, 3, 1, 2), params)
        background_uv_map = background_uv_map.permute(0, 2, 3, 1)

        # rescale flow values to conform to new image size
        scale_w = params['jitter size'][1] / self.frame_size[0]
        scale_h = params['jitter size'][0] / self.frame_size[1]

        flow[:, 0] *= scale_w
        flow[:, 1] *= scale_h
        background_flow[:, 0] *= scale_w
        background_flow[:, 1] *= scale_h
        input_tensor[:, :, 1] *= scale_w
        input_tensor[:, :, 2] *= scale_h

        # create jitter grid
        jitter_grid = self.initialize_jitter_grid()
        jitter_grid = self.apply_jitter_transform(jitter_grid, params)

        targets = {
            "rgb": rgb,
            "flow": flow,
            "masks": masks,
            "binary_masks": binary_masks,
            "flow_confidence": flow_conf
        }

        model_input = {
            "input_tensor": input_tensor,
            "spatiotemporal_noise": spatiotemporal_noise,
            "background_flow": background_flow,
            "background_uv_map": background_uv_map,
            "jitter_grid": jitter_grid,
            "index": torch.Tensor([idx, idx + 1]).long(),
        }

        return model_input, targets

    def __len__(self):
        return len(self.frame_iterator) - 2

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

    def initialize_jitter_grid(self):
        u = torch.linspace(-1, 1, steps=self.frame_size[0]).unsqueeze(0).repeat(self.frame_size[1], 1)
        v = torch.linspace(-1, 1, steps=self.frame_size[1]).unsqueeze(-1).repeat(1, self.frame_size[0])
        return torch.stack([u, v], 0).unsqueeze(0).repeat(2, 1, 1, 1)

    def apply_jitter_transform(self, input, params):
        """
        Apply jitter transform to the input.
        Input can either be of the form [T, L, C, H, W] or [T, C, H, W]
            if there is a L (layer) dimension T will always be 2

        Interpolate will process input of the form
            [T, C,   H, W]
            [L, C*T, H, W]
        """

        # specify whether or not there is a time dimension that needs to be stacked in channel dimension        
        rearrange_time_to_channel = len(input.shape) > 4
        
        tensor_size = params['jitter size'].tolist()
        crop_pos    = params['crop pos']
        crop_size   = params['crop size']

        # stack time dimension in channel dimension for `F.interpolate()`
        if rearrange_time_to_channel:
            input = torch.cat((input[0], input[1]), dim=-3)

        # if len(input.shape) < 4:
        #     data = F.interpolate(input.unsqueeze(0), size=tensor_size, mode=self.jitter_mode).squeeze(0)
        # else:
        data = F.interpolate(input, size=tensor_size, mode=self.jitter_mode)
        
        # separate time dimension
        if rearrange_time_to_channel:
            n_channel = input.size(-3)
            data = torch.stack((data[..., :n_channel // 2, :, :], 
                                data[..., n_channel // 2:, :, :]))

        data = data[..., crop_pos[0]:crop_pos[0] + crop_size[0], crop_pos[1]:crop_pos[1] + crop_size[1]]
        
        return data

    def get_jitter_parameters(self):
        """
        get random jitter parameters for the input.
        """

        width, height = self.frame_size

        if self.do_jitter:

            # get scale
            if np.random.uniform() > self.jitter_rate:
                scale = 1.
            else:
                scale = np.random.uniform(1, 1.25)
            
            # construct jitter
            jitter_size = (scale * np.array([height, width])).astype(np.int)
            start_h = np.random.randint(jitter_size[0] - height + 1)
            start_w = np.random.randint(jitter_size[1] - width  + 1)
        else:
            jitter_size = np.array([height, width])
            start_h = 0
            start_w = 0

        crop_pos  = np.array([start_h, start_w])
        crop_size = np.array([height,  width])

        params = {
            "jitter size": jitter_size, 
                "jitter size": jitter_size, 
            "jitter size": jitter_size, 
                "jitter size": jitter_size, 
            "jitter size": jitter_size, 
            "crop pos": crop_pos, 
                "crop pos": crop_pos, 
            "crop pos": crop_pos, 
                "crop pos": crop_pos, 
            "crop pos": crop_pos, 
            "crop size": crop_size
        }

        return params

    def prepare_image_dir(self, video_dir, out_dir):

        frame_paths = [frame for frame in sorted(listdir(video_dir))]
        
        for i, frame_path in enumerate(frame_paths):
            img = cv2.resize(cv2.imread(path.join(video_dir, frame_path)), self.frame_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(path.join(out_dir, f"{i:05}.jpg"), img)

    def load_composite_order(self, fp):
        self.composite_order = []
        
        if fp == None:
            create_new = True
        else:
            if path.exists(fp):
                with open(fp, "r") as f:
                    for line in f.readlines():
                        self.composite_order.append(tuple([int(frame_idx) for frame_idx in line.split(" ")]))
            else:
                create_new = True

        if create_new:
            for _ in range(len(self) + 1):
                self.composite_order.append(tuple([int(i+1) for i in range(self.mask_handler.N_objects)]))


class InputProcessor3D(object):
    def __init__(self,
                 video: str,
                 out_root: str,
                 initial_mask: Union[str, list],
                 composite_order_fp: Union[str, None],
                 propagation_model: str,
                 flow_model: str,
                 in_channels: int = 16,
                 noise_temporal_coarseness: int = 2,
                 device: str = "cuda",
                 timesteps: int = 2,
                 frame_size: tuple=(448, 256),
                 do_jitter: bool = True,
                 jitter_rate: float = 0.75
                ) -> None:
        super().__init__()

        self.device      = device
        self.frame_size  = frame_size
        self.in_channels = in_channels
        self.do_jitter   = do_jitter
        self.jitter_rate = jitter_rate
        self.jitter_mode = 'bilinear'
        self.timesteps   = timesteps

        if isinstance(initial_mask, str):
            self.N_objects = 1
        else:
            self.N_objects = len(initial_mask)

        self.N_layers = self.N_objects + 2

        self.img_dir   = path.join(out_root, "images")
        mask_dir       = path.join(out_root, "masks")
        flow_dir       = path.join(out_root, "flow")
        background_dir = path.join(out_root, "background")
        create_dirs(self.img_dir, mask_dir, flow_dir, background_dir)

        self.prepare_image_dir(video, self.img_dir)

        self.frame_iterator     = FrameIterator(self.img_dir, frame_size, device=self.device)
        self.mask_handler       = MaskHandler(video, mask_dir, initial_mask, frame_size, device=self.device, propagation_model=propagation_model)
        self.flow_handler       = FlowHandler(self.frame_iterator, self.mask_handler, flow_dir, raft_weights=flow_model, device=self.device)
        self.homography_handler = HomographyHandler(out_root, self.img_dir, path.join(flow_dir, "dynamics_mask"), self.device, frame_size)
        self.background_volume  = BackgroundVolume(background_dir, num_frames=len(self.frame_iterator), in_channels=in_channels, temporal_coarseness=noise_temporal_coarseness, frame_size=frame_size)       

        self.load_composite_order(composite_order_fp)
        
    def __getitem__(self, idx):
        """
        Get the optical flow input for the layer decompostion model

        Dimensions:
            T: time
            L: number of layers, always equal to N_objects + 2 (background)
            C: channel dimension, amount of channels in the input
            H: height of the image
            W: width of the image
           Nf: The number of frames in the video

        """
        # Get RGB input 
        # rgb: [C, T, H, W]
        rgb = self.frame_iterator[idx:idx + self.timesteps]

        # Get mask input
        # masks:        [L-2, C, T, H, W]
        # binary_masks: [L-2, C, T, H, W]
        masks, binary_masks = self.mask_handler[idx:idx + self.timesteps] 

        # Get optical flow input and confidence
        # flow:                 [C, T, H, W]
        # flow_conf:       [L-2,    T, H, W]
        # object_flow:     [L-2, C, T, H, W]
        # background_flow:      [C, T, H, W]
        flow, flow_conf, object_flow, _ = self.flow_handler[idx:idx + self.timesteps] 
        background_flow = self.homography_handler.frame_homography_to_flow(slice(idx, idx+self.timesteps)) 

        # Add background layers to masks
        masks = torch.cat((torch.zeros_like(masks[:, 0:1]).repeat(1, 2, 1, 1, 1), masks), dim=1)

        # Get spatial noise input
        # background_noise:  [1, C-3, T, H, W]
        # background_uv_map: [T, H, W, 2]
        background_noise = self.background_volume.spatial_noise_upsampled
        background_noise = background_noise.unsqueeze(0).unsqueeze(2).repeat(1, 1, self.timesteps, 1, 1) 

        background_uv_maps = []
        for frame_idx in range(idx, idx+self.timesteps):
            background_uv_maps.append(self.homography_handler.get_frame_uv(frame_idx))
        background_uv_map  = torch.stack(background_uv_maps)

        # Get spatiotemporal noise input
        # spatiotemporal_noise:   [C-3, Nf, H, W]
        # spatiotemporal_noise_t: [C-3,  T, H, W]
        spatiotemporal_noise    = self.background_volume.spatiotemporal_noise 

        # NOTE: time dimension is treated as batch dimension in grid_sample because we want to only sample in spatial dimensions
        spatiotemporal_noise_t = F.grid_sample(spatiotemporal_noise[:, idx:idx+self.timesteps].permute(1, 0, 2, 3), background_uv_map).permute(1, 0, 2, 3)

        # Construct query input
        zeros = torch.zeros((1, 3, self.timesteps, self.frame_size[1], self.frame_size[0]), dtype=torch.float32)  # [1, 3, T, H, W]
        static_background_input  = torch.cat((zeros, background_noise), dim=1)                                    # [1, C, T, H, W]
        dynamic_background_input = torch.cat((zeros, spatiotemporal_noise_t.unsqueeze(0)), dim=1)                 # [1, C, T, H, W]
        
        pids = binary_masks * (torch.Tensor(self.composite_order[idx]) + 1).view(self.N_layers - 2, 1, 1, 1, 1)             # [L-2, 1, T, H, W] 
        query_input = torch.cat((pids, object_flow, spatiotemporal_noise_t.repeat(self.N_layers - 2, 1, 1, 1, 1)), dim=1)   # [L-2, C, T, H, W]
        
        query_input = torch.cat((static_background_input, dynamic_background_input, query_input))                 # [L, C, T, H, W]

        # get full spatiotemporal noise input of the entire video
        spatiotemporal_noise = torch.cat((torch.zeros_like(spatiotemporal_noise[:3]), spatiotemporal_noise)) # [C, Nf, H, W]

        # get parameters for jitter
        params = self.get_jitter_parameters()
            
        # transform targets
        rgb       = self.apply_jitter_transform(rgb,       params)
        flow      = self.apply_jitter_transform(flow,      params)
        masks     = self.apply_jitter_transform(masks,     params)
        flow_conf = self.apply_jitter_transform(flow_conf, params)

        # transform inputs
        query_input      = self.apply_jitter_transform(query_input,    params)
        background_flow   = self.apply_jitter_transform(background_flow, params)
        background_uv_map = self.apply_jitter_transform(background_uv_map.permute(0, 3, 1, 2), params)
        background_uv_map = background_uv_map.permute(0, 2, 3, 1)

        # rescale flow values to conform to new image size
        scale_w = params['jitter size'][1] / self.frame_size[0]
        scale_h = params['jitter size'][0] / self.frame_size[1]

        flow[0] *= scale_w
        flow[1] *= scale_h
        background_flow[0] *= scale_w
        background_flow[1] *= scale_h
        query_input[:, 1] *= scale_w
        query_input[:, 2] *= scale_h

        # create jitter grid
        jitter_grid = self.initialize_jitter_grid()
        jitter_grid = self.apply_jitter_transform(jitter_grid, params)

        targets = {
            "rgb": rgb,
            "flow": flow,
            "masks": masks,
            "binary_masks": binary_masks,
            "flow_confidence": flow_conf
        }

        model_input = {
            "query_input": query_input,
            "spatiotemporal_noise": spatiotemporal_noise,
            "background_flow": background_flow,
            "background_uv_map": background_uv_map,
            "jitter_grid": jitter_grid,
            "index": torch.Tensor([idx, idx + 1]).long(),
        }

        return model_input, targets

    def __len__(self):
        return len(self.frame_iterator) - 2

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

    def initialize_jitter_grid(self):
        u = torch.linspace(-1, 1, steps=self.frame_size[0]).unsqueeze(0).repeat(self.frame_size[1], 1)
        v = torch.linspace(-1, 1, steps=self.frame_size[1]).unsqueeze(-1).repeat(1, self.frame_size[0])
        return torch.stack([u, v], 0).unsqueeze(0).repeat(2, 1, 1, 1)

    def apply_jitter_transform(self, input, params):
        """
        Apply jitter transform to the input.
        Input can either be of the form [L, C, T, H, W] or [C, T, H, W]

        Interpolate will process input of the form
            [C,   T, H, W]
            [L, C*T, H, W]
        """

        # specify whether or not there is a time dimension that needs to be stacked in channel dimension        
        rearrange_time_to_channel = len(input.shape) > 4
        
        tensor_size = params['jitter size'].tolist()
        crop_pos    = params['crop pos']
        crop_size   = params['crop size']

        # stack time dimension in channel dimension for `F.interpolate()`
        if rearrange_time_to_channel:
            L, C, T, H, W = input.shape
            input = input.view(L, C*T, H, W)

        data = F.interpolate(input, size=tensor_size, mode=self.jitter_mode)
        data = data[..., crop_pos[0]:crop_pos[0] + crop_size[0], crop_pos[1]:crop_pos[1] + crop_size[1]]
        
        # separate time dimension
        if rearrange_time_to_channel:
            data = data.view(L, C, T, H, W)

        return data

    def get_jitter_parameters(self):
        """
        get random jitter parameters for the input.
        """

        width, height = self.frame_size

        if self.do_jitter:

            # get scale
            if np.random.uniform() > self.jitter_rate:
                scale = 1.
            else:
                scale = np.random.uniform(1, 1.25)
            
            # construct jitter
            jitter_size = (scale * np.array([height, width])).astype(np.int)
            start_h = np.random.randint(jitter_size[0] - height + 1)
            start_w = np.random.randint(jitter_size[1] - width  + 1)
        else:
            jitter_size = np.array([height, width])
            start_h = 0
            start_w = 0

        crop_pos  = np.array([start_h, start_w])
        crop_size = np.array([height,  width])

        params = {
            "jitter size": jitter_size, 
                "jitter size": jitter_size, 
            "jitter size": jitter_size, 
                "jitter size": jitter_size, 
            "jitter size": jitter_size, 
            "crop pos": crop_pos, 
                "crop pos": crop_pos, 
            "crop pos": crop_pos, 
                "crop pos": crop_pos, 
            "crop pos": crop_pos, 
            "crop size": crop_size
        }

        return params

    def prepare_image_dir(self, video_dir, out_dir):

        frame_paths = [frame for frame in sorted(listdir(video_dir))]
        
        for i, frame_path in enumerate(frame_paths):
            img = cv2.resize(cv2.imread(path.join(video_dir, frame_path)), self.frame_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(path.join(out_dir, f"{i:05}.jpg"), img)

    def load_composite_order(self, fp):
        self.composite_order = []
        
        if fp == None:
            create_new = True
        else:
            if path.exists(fp):
                with open(fp, "r") as f:
                    for line in f.readlines():
                        self.composite_order.append(tuple([int(frame_idx) for frame_idx in line.split(" ")]))
            else:
                create_new = True

        if create_new:
            for _ in range(len(self) + 1):
                self.composite_order.append(tuple([int(i+1) for i in range(self.mask_handler.N_objects)]))


    ########################
    ###     Demo code    ###
    ########################

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