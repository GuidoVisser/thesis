from posix import listdir
import torch
import cv2
import numpy as np
import torch.nn.functional as F
from os import path
from typing import Union
from InputProcessing.depthHandler import DepthHandler

from utils.video_utils import opencv_folder_to_video
from utils.utils import create_dirs
from .backgroundVolume import BackgroundVolume
from .maskHandler import MaskHandler
from .flowHandler import FlowHandler
from .frameIterator import FrameIterator
from .homography import HomographyHandler


class InputProcessor(object):
    def __init__(self, args, do_jitter: bool = True,
                ) -> None:
        super().__init__()

        # initialize attributes
        self.timesteps                 = args.timesteps
        self.use_3d                    = not args.use_2d_loss_module
        self.frame_size                = (args.frame_width, args.frame_height)
        self.do_jitter                 = do_jitter
        self.jitter_rate               = args.jitter_rate
        self.jitter_mode               = 'bilinear'
        self.num_bg_layers             = 1 if args.no_static_background else 2

        # directories
        video = args.img_dir
        out_root = args.out_dir
        initial_mask = args.initial_mask

        if isinstance(initial_mask, str):
            self.N_objects = 1
        else:
            self.N_objects = len(initial_mask)

        self.N_layers = self.N_objects + self.num_bg_layers

        # create input directories
        self.out_root  = out_root

        img_dir        = path.join(out_root, "images")
        mask_dir       = path.join(out_root, "masks")
        flow_dir       = path.join(out_root, "flow")
        depth_dir      = path.join(out_root, "depth")
        background_dir = path.join(out_root, "background")
        create_dirs(img_dir, mask_dir, flow_dir, depth_dir, background_dir)

        # save resized ground truth frames in the input directory
        self._prepare_image_dir(video, img_dir)

        # create helper classes 
        #   These helpers prepare the mask propagation, homography estimation and optical flow calculation 
        #   at initialization and save the results for fast retrieval
        self.frame_iterator     = FrameIterator(img_dir, self.frame_size, device=args.device)
        self.mask_handler       = MaskHandler(video, mask_dir, initial_mask, self.frame_size, device=args.device, propagation_model=args.propagation_model)
        self.flow_handler       = FlowHandler(self.frame_iterator, self.mask_handler, flow_dir, raft_weights=args.flow_model, device=args.device, iters=50)
        self.homography_handler = HomographyHandler(out_root, img_dir, path.join(flow_dir, "dynamics_mask"), args.device, self.frame_size)
        self.depth_handler      = DepthHandler(img_dir, depth_dir, args.device, self.mask_handler, self.frame_size, args.depth_model)
        self.background_volume  = BackgroundVolume(background_dir, num_frames=len(self.frame_iterator), in_channels=args.in_channels, num_static_channels=args.num_static_channels, temporal_coarseness=args.noise_temporal_coarseness, frame_size=self.frame_size)        

        if args.model_type == "omnimatte":
            self.flow_handler.max_value = 1.

        # Load a custom compositing order if it's given, otherwise initialize a new one
        self._initialize_composite_order(args.composite_order)

        # placeholder for memory input
        self.memory_input = self.construct_memory_input(args.memory_input_type, args.shared_backbone)
        
    def __getitem__(self, idx):
        """
        Get the optical flow input for the layer decompostion model

        Dimensions:
            T: time
            L: number of layers, always equal to N_objects + 2 (background)
            C: channel dimension, amount of channels in the input
            H: height of the image
            W: width of the image
            F: The number of frames in the video
            b: The number of background layers (1 if only static, 2 if also dynamic)

        """
        # Get RGB input 
        # rgb: [C, T, H, W]
        rgb = self.frame_iterator[idx:idx + self.timesteps]

        # Get mask input
        # masks:        [L-b, C, T, H, W]
        # binary_masks: [L-b, C, T, H, W]
        masks, binary_masks = self.mask_handler[idx:idx + self.timesteps] 

        # Get optical flow input and confidence
        # flow:                 [C, T, H, W]
        # flow_conf:       [L-b,    T, H, W]
        # object_flow:     [L-b, C, T, H, W]
        # background_flow:      [C, T, H, W]
        flow, flow_conf, object_flow, _ = self.flow_handler[idx:idx + self.timesteps] 
        background_flow = self.homography_handler.frame_homography_to_flow(slice(idx, idx+self.timesteps)) 

        # Get depth input
        # depth:             [C, T, H, W]
        # object_depth: [L-b, C, T, H, W]
        depth, object_depth = self.depth_handler[idx:idx + self.timesteps]

        # Add background layers to masks
        masks = torch.cat((torch.zeros_like(masks[0:1]).repeat(self.num_bg_layers, 1, 1, 1, 1), masks))

        # Get spatial noise input if a separate static background is used
        # background_noise:  [1, C-4, T, H, W]
        # background_uv_map: [T, H, W, 2]
        if self.num_bg_layers == 2:
            background_noise = self.background_volume.spatial_noise_upsampled
            background_noise = background_noise.unsqueeze(0).unsqueeze(2).repeat(1, 1, self.timesteps, 1, 1) 

        background_uv_maps = []
        for frame_idx in range(idx, idx+self.timesteps):
            background_uv_maps.append(self.homography_handler.get_frame_uv(frame_idx))
        background_uv_map  = torch.stack(background_uv_maps)

        # Get spatiotemporal noise input
        # spatiotemporal_noise:   [C-4, F, H, W]
        # spatiotemporal_noise_t: [C-4, T, H, W]
        spatiotemporal_noise = self.background_volume.spatiotemporal_noise 

        # NOTE: time dimension is treated as batch dimension in grid_sample because we want to only sample in spatial dimensions
        spatiotemporal_noise_t = F.grid_sample(spatiotemporal_noise[:, idx:idx+self.timesteps].permute(1, 0, 2, 3), background_uv_map).permute(1, 0, 2, 3)

        ### Temp code ###

        # img_t = spatiotemporal_noise_t[:3, 0].permute(1, 2, 0).cpu().numpy() * .25 + 0.75
        # img = spatiotemporal_noise[:3, idx].permute(1, 2, 0).cpu().numpy() * .25 + 0.75

        # uv = background_uv_map[0].clone().cpu().numpy()
        # corners = [uv[0, 0], uv[0, -1], uv[-1, -1], uv[-1, 0]]

        # corners = np.array([(c * .5 + .5) * np.array([448, 256]) for c in corners]).reshape((-1, 1, 2)).astype(np.int32)

        # color = (1, 0, 0)
        # thickness = 3

        # img = cv2.polylines(np.ascontiguousarray(img), [corners], True, color, thickness, cv2.LINE_AA)

        # # color = (255, 0, 0)
        # # thickness = 3

        # # img = cv2.polylines(np.ascontiguousarray(img), [corners], True, color, thickness)

        # cv2.imwrite(path.join(self.out_root, "st_noise.png"), img * 255)
        # cv2.imwrite(path.join(self.out_root, "st_sampled_noise.png"), img_t * 255)

        #################

        # add uniform sampling in time dimension to background uv map if necessary
        if self.use_3d:
            t = torch.linspace(-1, 1, self.timesteps).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, self.frame_size[1], self.frame_size[0], 1)

            background_uv_map = torch.cat((background_uv_map, t), dim=-1)

        # Construct query input        
        pids = binary_masks * (torch.Tensor(self.composite_order[idx]) + 1).view(self.N_layers - self.num_bg_layers, 1, 1, 1, 1)                         # [L-b, 1, T, H, W] 
        query_input = torch.cat((pids, object_depth, object_flow, spatiotemporal_noise_t.repeat(self.N_layers - self.num_bg_layers, 1, 1, 1, 1)), dim=1) # [L-b, C, T, H, W]

        zeros = torch.zeros((1, 4, self.timesteps, self.frame_size[1], self.frame_size[0]), dtype=torch.float32)  # [1, 4, T, H, W]
        if self.num_bg_layers == 2:
            static_background_input  = torch.cat((zeros, background_noise), dim=1)                                # [1, C, T, H, W]
            dynamic_background_input = torch.cat((zeros, spatiotemporal_noise_t.unsqueeze(0)), dim=1)             # [1, C, T, H, W]
            background_input = torch.cat((static_background_input, dynamic_background_input))
        else:
            background_input = torch.cat((zeros, spatiotemporal_noise_t.unsqueeze(0)), dim=1)                     # [1, C, T, H, W]            

        query_input = torch.cat((background_input, query_input)) # [L, C, T, H, W]

        # get parameters for jitter
        params = self.get_jitter_parameters()
            
        # transform targets
        rgb       = self.apply_jitter_transform(rgb,       params)
        flow      = self.apply_jitter_transform(flow,      params)
        depth     = self.apply_jitter_transform(depth,     params)
        masks     = self.apply_jitter_transform(masks,     params)
        flow_conf = self.apply_jitter_transform(flow_conf, params)

        # transform inputs
        query_input       = self.apply_jitter_transform(query_input,     params)
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

        # create jitter grid for background offset and brightness scaling
        if self.use_3d:
            jitter_grid = self.initialize_3d_jitter_grid()
        else:
            jitter_grid = self.initialize_2d_jitter_grid()
        jitter_grid = self.apply_jitter_transform(jitter_grid, params)

        targets = {
            "rgb": rgb,
            "flow": flow,
            "depth": depth,
            "masks": masks,
            "binary_masks": binary_masks,
            "flow_confidence": flow_conf
        }

        model_input = {
            "query_input": query_input,
            "memory_input": self.memory_input,
            "background_flow": background_flow,
            "background_uv_map": background_uv_map,
            "jitter_grid": jitter_grid,
            "index": torch.arange(idx, idx + self.timesteps).long(),
        }

        return model_input, targets

    def __len__(self):
        return len(self.frame_iterator) - self.timesteps

    def construct_memory_input(self, memory_input_type: str, shared_backbone: bool) -> torch.Tensor:
        """
        Construct the memory input of the video

        There are four configurations, sometimes with a special version supporting a shared memory and query encoder backbone:
            1. Only rgb (shared backbone not possible)
            2. rgb + depth + flow (shared backbone not possible)
            3. Only noise
            4. noise + depth + flow

        Args:
            memory_input_config (dict): dictionary containing the configuration for the memory input

        Returns:
            memory_input (torch.Tensor): Tensor used as memory input throughout the full training
        """

        if memory_input_type == "rgb":
            memory_input = self.frame_iterator[:len(self.flow_handler)]        # [3, F-1, H, W]
        
        elif memory_input_type == "rgb_flow_depth":
            rgb_input           = self.frame_iterator[:len(self.flow_handler)]         # [3, F-1, H, W]
            flow_input, _, _, _ = self.flow_handler[:len(self.flow_handler)]           # [2, F-1, H, W]
            depth_input, _      = self.depth_handler[:len(self.flow_handler)]          # [1, F-1, H, W]
            memory_input        = torch.cat((rgb_input, flow_input, depth_input))      # [6, F-1, H, W]
        
        elif memory_input_type == "noise_flow_depth":
            noise_input         = self.background_volume.spatiotemporal_noise[:, :-1] # [C-4, F-1, H, W]
            depth_input, _      = self.depth_handler[:len(self.flow_handler)]         # [  1, F-1, H, W]
            flow_input, _, _, _ = self.flow_handler[:len(self.flow_handler)]          # [  2, F-1, H, W]
        
            if shared_backbone:
                mask_padding = torch.zeros_like(noise_input[:1])                               # [  1, F-1, H, W]
                memory_input = torch.cat((mask_padding, depth_input, flow_input, noise_input)) # [C,   F-1, H, W]
            else:
                memory_input = torch.cat((depth_input, flow_input, noise_input))               # [C,   F-1, H, W]
        
        elif memory_input_type == "noise":
            memory_input  = self.background_volume.spatiotemporal_noise # [C-4, F-1, H, W]
        
            if shared_backbone:
                padding = torch.zeros_like(memory_input[:4])            # [4, F-1, H, W]
                memory_input = torch.cat((padding, memory_input))       # [C, F-1, H, W]

        return memory_input

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

    def initialize_2d_jitter_grid(self):
        u = torch.linspace(-1, 1, steps=self.frame_size[0]).unsqueeze(0).repeat(self.frame_size[1], 1)
        v = torch.linspace(-1, 1, steps=self.frame_size[1]).unsqueeze(-1).repeat(1, self.frame_size[0])
        return torch.stack([u, v], 0).unsqueeze(0).repeat(2, 1, 1, 1)

    def initialize_3d_jitter_grid(self):
        t = torch.linspace(-1, 1, steps=self.timesteps).unsqueeze(1).unsqueeze(1).repeat(1, self.frame_size[1], self.frame_size[0])
        u = torch.linspace(-1, 1, steps=self.frame_size[0]).unsqueeze(0).unsqueeze(0).repeat(self.timesteps, self.frame_size[1], 1)
        v = torch.linspace(-1, 1, steps=self.frame_size[1]).unsqueeze(0).unsqueeze(-1).repeat(self.timesteps, 1, self.frame_size[0])
        
        return torch.stack([u, v, t], 0)

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
            "crop pos": crop_pos, 
            "crop size": crop_size
        }

        return params

    def _prepare_image_dir(self, video_dir, out_dir):
        """
        Prepare an image directory with the frames at the correct resolution
        """
        frame_paths = [frame for frame in sorted(listdir(video_dir))]
        
        for i, frame_path in enumerate(frame_paths):
            img = cv2.resize(cv2.imread(path.join(video_dir, frame_path)), self.frame_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(path.join(out_dir, f"{i:05}.jpg"), img)

    def _initialize_composite_order(self, fp: str) -> None:
        """
        Initialize a compositing order for each frame in the video

        If a filepath is given, a custom compositing order is loaded, otherwise the order of each frame
        is initialized in the same order
        """
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