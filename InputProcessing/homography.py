import torch
import numpy as np
import cv2
from os import path, listdir, remove
from typing import Tuple, Union
import imageio

from models.third_party.SuperGlue.models.utils import frame2tensor

from .utils.utils import get_translation_matrix, homogeneous_2d_transform
from .utils.feature_matching import extract_and_match_features, get_matching_coordinates, get_model_config
from .utils.remove_foreground import remove_foreground_features

class HomographyHandler(object):
    def __init__(self,
                 savepath: str,
                 image_dir: str,
                 mask_dir: list,
                 device: str = "cuda",
                 frame_size: list = [854, 480],
                 interval: int = 1) -> None:
        super().__init__()
        self.device = device

        self.mask_dirs = [path.join(mask_dir, dir) 
                               for dir 
                               in sorted(listdir(mask_dir))]
        self.mask_dir = mask_dir

        self.frame_sequence = [path.join(image_dir, frame) 
                               for frame 
                               in sorted(listdir(image_dir))]

        self.frame_size = frame_size
        self.interval = interval

        homography_path = path.join(savepath, "homographies.txt")
        if path.exists(homography_path):
            self.load_homography(homography_path)
        else:
            self.homographies, self.origin_size = self.get_homographies()
            self.xmin, self.ymin, self.xmax, self.ymax = self.get_outer_borders()
            self.save_homography(homography_path)

        self.homography_demo(savepath)

    def load_homography(self, filepath):
        with open(filepath, "r") as f:
            homography_data = f.readlines()

        self.origin_size = [int(item) 
                            for item 
                            in homography_data[0].rstrip().split(" ")]
        self.xmin, self.xmax, self.ymin, self.ymax = [int(item)
                                                      for item 
                                                      in homography_data[1].rstrip().split(" ")]
        homographies = homography_data[2:]
        self.homographies = [torch.from_numpy(np.array(line.rstrip().split(" ")).astype(np.float32).reshape(3, 3))
                             for line
                             in homographies]

    def save_homography(self, filepath):
        
        try:
            with open(filepath, "a") as f:
                f.write(" ".join([str(item) for item in self.origin_size]) + "\n")
                f.write(" ".join(str(item) for item in [self.xmin, self.xmax, self.ymin, self.ymax]) + "\n")
                for homography in self.homographies:
                    f.write(" ".join([str(item.item()) for item in list(homography.reshape(-1))]) + "\n")
        except:
            remove(filepath)
            raise ValueError


    def get_outer_borders(self) -> Tuple[float, float, float, float]:
        """
        Get the outer borders of the frames when they are aligned using homographies

        Returns:
            xmin (float)
            ymin (float)
            xmax (float)
            ymax (float)
        """

        # Get the base corners of a frame
        w, h = self.frame_size
        corners = np.float32([[0, 0], 
                              [0, h],
                              [w, h], 
                              [w, 0]]).reshape(-1, 1, 2)  
        
        # get the corners of all warped frames
        all_corners = [corners]
        for homography in self.homographies:
            homography = homography.numpy()
            warped_corners = cv2.perspectiveTransform(corners, homography)
            all_corners.append(warped_corners)
            corners = warped_corners
        all_corners = np.concatenate(all_corners, axis=0)

        # find the extreme values in the corners
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        return xmin, ymin, xmax, ymax

    def get_frame_uv(self, frame_idx: int) -> torch.Tensor:
        """
        Get a UV map for the homography warping of a frame that can be used by torch.grid_sample()
        """
        w, h = self.frame_size

        u  = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
        v  = torch.linspace(0, 1, steps=h).unsqueeze(-1).repeat(1, w)
        uv = torch.stack((u, v))

        uv[0] *= self.origin_size[0]
        uv[1] *= self.origin_size[1]

        uv = uv.reshape(2, -1)
        homography = self.calculate_homography_between_two_frames(0, frame_idx)

        [xt, yt] = homogeneous_2d_transform(uv[0], uv[1], homography)
        xt -= self.xmin
        xt /= float(self.xmax - self.xmin)
        yt -= self.ymin
        yt /= float(self.ymax - self.ymin)

        uv = torch.stack([xt.reshape(h, w), yt.reshape(h, w)]).permute(1, 2, 0) * 2 - 1

        return uv 

    def extract_homography(self, coords: list) -> list:
        """
        Find the homography operation between a set of pairs of frames with known feature matches

        Args:
            coords (list): list of corresponding coordinates of keypoints in pairs of frames
        
        Returns:
            homographies (list[np.array]): list of homoghraphy operations for every frame pair
        """

        homographies = []
        for (coords0, coords1) in coords:
            homography, _ = cv2.findHomography(coords0, coords1, cv2.RANSAC)
            homographies.append(torch.from_numpy(homography).float())
            
        return homographies

    def calculate_homography_between_two_frames(self, source: int, target: int) -> np.array:
        """
        Get the homography that maps the perspective from a source frame onto a target frame

        Homography is transative meaning that:
            H_1 (H_2 F) = (H_1 H_2) F

        so for any source, target pair (a, b) we can sequentially apply the homography matrices like
            H* = H_a H_a+1 H_a+2 ... H_b-1 H_b

        Args:
            source (int): index of source frame
            target (int): index of target frame
        
        Returns:
            homography (np.array): 3x3 homography mapping from source to target perspective
        """

        # create a sequence of homographies defined by the source and target index
        if target > source:
            homography_sequence = self.homographies[source:target]
        elif target < source:
            homography_sequence = self.homographies[target:source]
        else:
            return torch.eye(3)

        # sequentially apply homographies, leveraging the transative property of homography
        homography = homography_sequence[0]
        for i in range(1, len(homography_sequence)):
            homography = homography_sequence[i] @ homography
    
        # if target comes before source in the sequence take the inverse homography
        if target < source:
            homography = torch.inverse(homography)

        return homography
    
    def get_homographies(self):
        """
        create a list of homographies between the subsequent frames in the sequence
        
        For a frame sequence of length N, this function will return N-1 3x3 homography matrices

        Returns:
            homographies (list[np.arrray])
        """
        # get frames and frame tensors
        _, frame_tensors = self.get_images()
        
        origin_size = list(frame_tensors[0].shape[2:4])
        origin_size.reverse()
        
        # get masks
        masks = self.get_masks()

        # extract features using SuperPoint and match the features using SuperGlue
        feature_matches = extract_and_match_features(frame_tensors, self.device, get_model_config())
        
        # get the coordinates of matching features for RANSAC
        coords = get_matching_coordinates(feature_matches)

        # remove foreground features for cleaner homography estimation
        coords = remove_foreground_features(coords, masks)
        
        # apply RANSAC and get homographies from coordinates 
        homographies = self.extract_homography(coords)

        return homographies, origin_size

    def frame_homography_to_flow(self, frame_index: Union[int, slice]):
        """
        Calculate a flow field that 
        """
        w, h = self.frame_size    

        ramp_u = torch.linspace(0, self.origin_size[0], steps=w).unsqueeze(0).repeat(h, 1)
        ramp_v = torch.linspace(0, self.origin_size[1], steps=h).unsqueeze(-1).repeat(1, w)
        ramp_ = torch.stack([ramp_u, ramp_v], 0)
        ramp = ramp_.reshape(2, -1)


        if isinstance(frame_index, slice):
            flows = []
            for idx in range(frame_index.start, frame_index.stop):
                H = self.calculate_homography_between_two_frames(idx + 1, idx)

                # apply homography
                [xt, yt] = homogeneous_2d_transform(ramp[0], ramp[1], H)

                # restore shape
                flow = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)
                flow -= ramp_

                # scale from world to image space
                flow[0] *= w / self.origin_size[0]
                flow[1] *= h / self.origin_size[1]
            
                flows.append(flow)

            flow = torch.stack(flows, dim=-3)
        else:
            H = self.calculate_homography_between_two_frames(frame_index + 1, frame_index)
            
            # apply homography
            [xt, yt] = homogeneous_2d_transform(ramp[0], ramp[1], H)
            
            # restore shape
            flow = torch.stack([xt.reshape(h, w), yt.reshape(h, w)], 0)
            flow -= ramp_
            
            # scale from world to image space
            flow[0] *= w / self.origin_size[0]
            flow[1] *= h / self.origin_size[1]

        return flow

    def align_frames(self, frames: list, indices: Union[list, None] = None) -> list:
        """
        Expand, warp and align a set of frames such that they overlay each other correctly 
        using the respective homographies

        Args:
            frames  (list[np.array]): list of np.arrays containing the frames
            indices (Union[list, None]): list of frame indices.
                                         If it is None range (0, N_frames) is used
        """
        
        Ht = get_translation_matrix(-self.xmin, -self.ymin)

        warped_frames = []

        if indices is not None:
            if len(indices < 2):
                raise ValueError("At least two frames need to be provided")
            initial_idx = indices[0]
            frame_range = indices[1:]
        else:
            initial_idx = 0
            frame_range = range(1, len(frames))

        for i in frame_range:
            homography = self.calculate_homography_between_two_frames(initial_idx, i)
            warped_frames.append(cv2.warpPerspective(frames[i], Ht @ homography.numpy(), (self.xmax - self.xmin, self.ymax - self.ymin)))

        source_padded = np.zeros((self.ymax-self.ymin, self.xmax-self.xmin, 3), dtype=np.uint8)
        source_padded[-self.ymin:self.frame_size[1]-self.ymin, -self.xmin:self.frame_size[0]-self.xmin] = frames[0]

        warped_frames.insert(0, source_padded)

        return warped_frames

    def get_images(self) -> list: 
        """
        Load all images in the directory with a given interval between them
        Images are converted to grayscale and then to torch.Tensors for SuperGlue

        Args:
            img_dir  (str): path to directory where images are stored 
            device   (str): device for the SuperGlue model
            resize   (list[int]): size to which frames are rescaled before being processed
            interval (int): interval between images
        
        Returns:
            frames        (list[np.array]) = list of frames
            frame_tensors (list[np.array]) = list of tensors usable by SuperGlue model
        """

        # loop through file paths and process images
        frames, frame_tensors = [], []
        for frame_path in self.frame_sequence:

            # load frameremove_foreground_features
            frame = cv2.imread(frame_path)

            # convert to grayscale for SuperGlue
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float32')

            # add image and image tensor to output
            frames.append(frame)
            frame_tensors.append(frame2tensor(frame_gray, self.device))

        return frames, frame_tensors

    def get_masks(self, binary_threshold: float = 0.5) -> list:
        """
        Load all object masks of the frames that are being considered and construct a single global
        mask for all foreground objects 

        Args:
            binary_threshold (float): threshold for a pixel being masked by a non-binary mask
        
        Returns:
            masks (list): list of combined masks
        """


        masks = []
        for idx, mask_path in enumerate(sorted(listdir(self.mask_dir))):
            if idx % self.interval == 0:
                masks.append(cv2.imread(path.join(self.mask_dir, mask_path), cv2.IMREAD_GRAYSCALE) / 255.)

        return masks

    def _initialize_masks(self, n_masks: int, mask_shape: list) -> list:
        """
        Initialize a list of masks with the correct size. All values initialized to zero

        The directory with the relevant masks are passed in order to inform the amount of masks and the 
        size of the masks that will be initialized

        Args:
            n_masks  (int): number of masks that need to be initialized
            interval (int): shape of the masks

        Returns:
            masks (list[np.array]): list of np.arrays with all zero values
        """
        return [np.zeros(mask_shape) for _ in range(n_masks)]

    def homography_demo(self, savepath):
        """
        Create a demo of the camera stabilization results
        """
        frames = [cv2.imread(frame_path) for frame_path in self.frame_sequence]
        aligned_frames = self.align_frames(frames)
        img_array = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in aligned_frames]
        img_array = np.stack(img_array)
        video_path = path.join(savepath, "homography_demo.gif")
        imageio.mimsave(video_path, img_array, format="GIF", fps=25)