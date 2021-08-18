from InputProcessing.homography import HomographyHandler
from os import path, listdir
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union

from .utils.utils import get_scale_matrix, get_translation_matrix

class BackgroundVolume(object):

    def __init__(self,
                 save_dir: str,
                 in_channels: int = 16,
                 frame_size: list = [864, 480]) -> None:
        super().__init__()
        self.save_dir  = save_dir

        self.spatial_noise = torch.randn(1, in_channels - 3, frame_size[1] // 16, frame_size[0] // 16)
        self.spatial_noise_upsampled = F.interpolate(self.spatial_noise, (frame_size[1], frame_size[0]), mode='bilinear')

        torch.save(self.spatial_noise, path.join(save_dir, "spatial_noise.pth"))
        torch.save(self.spatial_noise_upsampled, path.join(save_dir, "spatial_noise_upsampled.pth"))
        
        
        # self.frame_sequence = [path.join(image_dir, frame) 
        #                        for i, frame 
        #                        in enumerate(sorted(listdir(image_dir))) 
        #                        if i % interval == 0]
        
        # self.mask_dirs = [path.join(mask_dir, dir) for dir in sorted(listdir(mask_dir))]
        # self.interval   = interval

        # self.homography_handler = homography_handler
        

    # def get_frame_noise(self, frame_idx: int) -> np.array:
    #     """
    #     Get a perspective projection of the static spatial noise into the given frame

    #     Args:
    #         frame_idx (int): index of the relevant frame 
        
    #     Returns:
    #         warped_noise (np.array): the spatial noise warped into the selected frames perspective
    #     """

    #     homography = self.homography_handler.calculate_homography_between_two_frames(0, frame_idx)
    #     xmin = self.homography_handler.xmin
    #     ymin = self.homography_handler.ymin
    #     xmax = self.homography_handler.xmax
    #     ymax = self.homography_handler.ymax

    #     Ht = get_translation_matrix(-xmin, -ymin)
    #     Hs = get_scale_matrix(xmax - xmin, 
    #                           ymax - ymin, 
    #                           self.spatial_noise.shape[1], 
    #                           self.spatial_noise.shape[0])
    #     H = Hs @ Ht

    #     transformation_matrix = np.linalg.inv(H @ homography)

    #     warped_noise = cv2.warpPerspective(self.spatial_noise, transformation_matrix, self.frame_size, borderMode=cv2.BORDER_REPLICATE)
    #     return warped_noise


    # def get_images(self) -> list: 
    #     """
    #     Load all images in the directory with a given interval between them
    #     Images are converted to grayscale and then to torch.Tensors for SuperGlue

    #     Args:
    #         img_dir  (str): path to directory where images are stored 
    #         device   (str): device for the SuperGlue model
    #         resize   (list[int]): size to which frames are rescaled before being processed
    #         interval (int): interval between images
        
    #     Returns:
    #         frames        (list[np.array]) = list of frames
    #         frame_tensors (list[np.array]) = list of tensors usable by SuperGlue model
    #     """

    #     # loop through file paths and process images
    #     frames, frame_tensors = [], []
    #     for frame_path in self.frame_sequence:

    #         # load frameremove_foreground_features
    #         frame = cv2.imread(frame_path)

    #         # resize image
    #         w, h         = frame.shape[1], frame.shape[0]
    #         w_new, h_new = process_resize(w, h, self.frame_size)
    #         frame        = cv2.resize(frame, (w_new, h_new))
            
    #         # convert to grayscale for SuperGlue
    #         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float32')

    #         # add image and image tensor to output
    #         frames.append(frame)
    #         frame_tensors.append(frame2tensor(frame_gray, self.device))

    #     return frames, frame_tensors

    # def get_masks(self, binary_threshold: float = 0.9) -> list:
    #     """
    #     Load all object masks of the frames that are being considered and construct a single global
    #     mask for all foreground objects 

    #     Args:
    #         binary_threshold (float): threshold for a pixel being masked by a non-binary mask
        
    #     Returns:
    #         masks (list): list of combined masks
    #     """

    #     n_masks = len(listdir(self.mask_dirs[0]))//self.interval
    #     masks   = self._initialize_masks(n_masks, (self.frame_size[1], self.frame_size[0]))
    #     for dir in self.mask_dirs:

    #         mask_paths = [path.join(dir, mask) for mask in sorted(listdir(dir))]
    #         for i, mask_path in enumerate(mask_paths):

    #             # skip iteration if it is not a valid iteration according to the interval
    #             if i % self.interval != 0:
    #                 continue

    #             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    #             # resize mask
    #             w, h         = mask.shape[1], mask.shape[0]
    #             w_new, h_new = process_resize(w, h, self.frame_size)
    #             mask         = cv2.resize(mask, (w_new, h_new))
                
    #             # ensure that mask is binary
    #             _, mask = cv2.threshold(mask, binary_threshold, 1, cv2.THRESH_BINARY)

    #             masks[i//self.interval] = np.add(masks[i//self.interval], mask)

    #     for mask in masks:
    #         np.minimum(mask, np.ones(mask.shape), mask)

    #     return masks

    # def construct_frame_volume(self, frame_idx: int):
    #     """
    #     Construct a background volume for a single frame

    #     Args:
    #         frame_idx (int): 
    #     """

    #     frames, _ = self.get_images()
    #     masks     = self.get_masks()

    #     frames = mask_out_foreground(frames, masks)

    #     noise = self.get_frame_noise(frame_idx)

    #     aligned_frames = []
    #     for i, frame in enumerate(frames):
           
    #         if i == frame_idx:
    #             aligned_frames.append(frame)
    #         else:
    #             homography = self.calculate_homography_between_two_frames(frame_idx, i)
    #             aligned_frames.append(cv2.warpPerspective(frame, homography, (self.frame_size[0], self.frame_size[1])))

    #     aligned_frames = self.add_background_noise(aligned_frames, noise)

    #     # save results if path is specified
    #     if self.save_dir is not None:

    #         for i, frame in enumerate(aligned_frames):
    #             cv2.imwrite(path.join(self.save_dir, f"{i*self.interval:05}.jpg"), frame)

    #     return aligned_frames

    # def construct_full_volume(self) -> Tuple[list, np.array]:
    #     """
    #     construct a background volume:
    #         - create a global image window in which all frames can fit when aligned
    #         - remove foreground objects from the frames
    #         - warp and align all frames using the homographies
    #         - add a static spatial noise to the empty regions
    #         - save aligned frames             
    #     """

    #     # get frames and frame tensors
    #     frames, _ = self.get_images()
        
    #     # get masks
    #     masks = self.get_masks()

    #     # mask out the foreground objects
    #     frames = mask_out_foreground(frames, masks)

    #     # align all frames
    #     aligned_frames = self.align_frames(frames)

    #     # add static noise to empty regions
    #     aligned_frames = self.add_background_noise(aligned_frames, self.demo_spatial_noise)

    #     # save results if path is specified
    #     if self.save_dir is not None:

    #         for i, frame in enumerate(aligned_frames):
    #             cv2.imwrite(path.join(self.save_dir, f"{i*self.interval:05}.jpg"), frame)

    #     return aligned_frames

    # def add_background_noise(self, frames: list, noise: np.array) -> list:
    #     """
    #     Add a static spatial noise to the empty regions of the warped frames

    #     Args:
    #         frames (list[np.array]): list of frames
    #         noise  (np.array): spatial noise, should be of the same dimensions as the frames

    #     Returns:
    #         processed_frames (list): list of frames with the random noise added
    #     """
    #     processed_frames = []
    #     for frame in frames:
    #         assert frame.shape == noise.shape, f"The shape of the image and the noise should be the same. Got {frame.shape} (image) and {noise.shape} (noise)"

    #         # NOTE fix the threshold
    #         _, mask = cv2.threshold(frame, 0, 1, cv2.THRESH_BINARY_INV)
    #         frame = frame + noise * mask
    #         processed_frames.append(frame)
        
    #     return processed_frames

    # def _initialize_masks(self, n_masks: int, mask_shape: list) -> list:
    #     """
    #     Initialize a list of masks with the correct size. All values initialized to zero

    #     The directory with the relevant masks are passed in order to inform the amount of masks and the 
    #     size of the masks that will be initialized

    #     Args:
    #         n_masks  (int): number of masks that need to be initialized
    #         interval (int): shape of the masks

    #     Returns:
    #         masks (list[np.array]): list of np.arrays with all zero values
    #     """
    #     return [np.zeros(mask_shape) for _ in range(n_masks)]

    ###############################
    ###        Debugging        ###
    ###############################

    # def create_demo_noise(self):
    #     """
    #     Create a demo spatial noise to more easily investigate properties of the homographies
    #     """
    #     w = self.xmax - self.xmin
    #     h = self.ymax - self.ymin
    #     w, h = self.frame_size[0], self.frame_size[1]

    #     noise_x = np.stack([np.linspace(np.arange(w)/w, np.arange(w)/w, h)]*2, axis=2)
    #     noise_y = np.expand_dims(np.transpose(np.linspace(np.arange(h)/h, np.arange(h)/h, w)), 2)

    #     self.demo_spatial_noise = np.concatenate([noise_x, noise_y], axis=2)
    
    # def draw_frame_border(self, image, Ht, homography):
    #     w, h = self.frame_size
    #     rect = np.float32([[0, 0], 
    #                        [0, h],
    #                        [w, h], 
    #                        [w, 0]]).reshape(-1, 1, 2) 
    #     rect = cv2.perspectiveTransform(rect, Ht@homography)+0.5
    #     rect = rect.astype(np.int32)
        
    #     image = cv2.polylines(image, [rect], True, (0,255,0), 8)
    #     return image