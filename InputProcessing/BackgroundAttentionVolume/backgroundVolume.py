from os import path, listdir
import cv2
import numpy as np
from typing import Union, Tuple

from models.SuperGlue.models.utils import process_resize, frame2tensor
from utils.utils import create_dir

from .feature_matching import get_model_config, extract_and_match_features, get_matching_coordinates
from .homography import extract_homography
from .remove_foreground import remove_foreground_features, mask_out_foreground
from .utils import get_translation_matrix


class BackgroundVolume(object):

    def __init__(self,
                 image_dir: str,
                 mask_dirs: Union[str, list],
                 device: str,
                 interval: int = 1,
                 save_dir: Union[str, None] = None,
                 frame_size: list = [864, 480]) -> None:
        super().__init__()
        self.frame_sequence = [path.join(image_dir, frame) 
                               for i, frame 
                               in enumerate(sorted(listdir(image_dir))) 
                               if i % interval == 0]
        
        if isinstance(mask_dirs, str):
            self.mask_dirs = [mask_dirs]
        else:
            self.mask_dirs = mask_dirs
        
        self.save_dir = save_dir
        if self.save_dir is not None:
            create_dir(self.save_dir)

        self.interval = interval
        self.device = device
        self.frame_size = frame_size

        self.homographies = self.get_homographies()
        self.xmin, self.ymin, self.xmax, self.ymax = self.get_outer_borders()

        try:
            self.spatial_noise = np.random.uniform(low=0., high=255., size=(self.ymax - self.ymin, self.xmax - self.xmin, 3))
        except Exception as e:
            print(e)
        
    def create_demo_noise(self):
        """
        Create a demo spatial noise to more easily investigate properties of the homographies
        """
        w = self.xmax - self.xmin
        h = self.ymax - self.ymin
        self.spatial_noise = np.stack([np.linspace(np.arange(w)/w, np.arange(w)/w, h)]*3, axis=2)

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
            return np.identity(3)

        # sequentially apply homographies, leveraging the transative property of homography
        homography = homography_sequence[0]
        for i in range(1, len(homography_sequence)):
            homography = homography_sequence[i] @ homography
    
        # if target comes before source in the sequence take the inverse homography
        if target < source:
            homography = np.linalg.inv(homography)

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
        
        # get masks
        masks = self.get_masks()

        # extract features using SuperPoint and match the features using SuperGlue
        feature_matches = extract_and_match_features(frame_tensors, self.device, get_model_config())
        
        # get the coordinates of matching features for RANSAC
        coords = get_matching_coordinates(feature_matches)

        # remove foreground features for cleaner homography estimation
        coords = remove_foreground_features(coords, masks)
        
        # apply RANSAC and get homographies from coordinates 
        homographies = extract_homography(coords)

        return homographies

    def get_images(self) -> list: 
        """
        Load all images in the directory with a given interval between them
        Images are converted to grayscale and then to torch.Tensors for SuperGlue

        Args:
            img_dir (str): path to directory where images are stored 
            device (str): device for the SuperGlue model
            resize (list[int]): size to which frames are rescaled before being processed
            interval (int): interval between images
        
        Returns:
            frames (list[np.array]) = list of frames
            frame_tensors (list[np.array]) = list of tensors usable by SuperGlue model
        """

        # loop through file paths and process images
        frames, frame_tensors = [], []
        for frame_path in self.frame_sequence:

            # load frame
            frame = cv2.imread(frame_path)

            # resize image
            w, h = frame.shape[1], frame.shape[0]
            w_new, h_new = process_resize(w, h, self.frame_size)
            frame = cv2.resize(frame, (w_new, h_new))
            
            # convert to grayscale for SuperGlue
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float32')

            # add image and image tensor to output
            frames.append(frame)
            frame_tensors.append(frame2tensor(frame_gray, self.device))

        return frames, frame_tensors

    def get_masks(self, binary_threshold: float = 0.9) -> list:
        """
        Load all object masks of the frames that are being considered and construct a single global
        mask for all foreground objects 

        Args:
            binary_threshold (float): threshold for a pixel being masked by a non-binary mask
        
        Returns:
            masks (list): list of combined masks
        """

        n_masks = len(listdir(self.mask_dirs[0]))//self.interval
        masks = self._initialize_masks(n_masks, (self.frame_size[1], self.frame_size[0]))
        for dir in self.mask_dirs:

            mask_paths = [path.join(dir, mask) for mask in sorted(listdir(dir))]
            for i, mask_path in enumerate(mask_paths):

                # skip iteration if it is not a valid iteration according to the interval
                if i % self.interval != 0:
                    continue

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # resize mask
                w, h = mask.shape[1], mask.shape[0]
                w_new, h_new = process_resize(w, h, self.frame_size)
                mask = cv2.resize(mask, (w_new, h_new))
                
                # ensure that mask is binary
                _, mask = cv2.threshold(mask, binary_threshold, 1, cv2.THRESH_BINARY)

                masks[i//self.interval] = np.add(masks[i//self.interval], mask)

        for mask in masks:
            np.minimum(mask, np.ones(mask.shape), mask)

        return masks

    def construct_frame_volume(self, frame_idx: int):
        """
        Construct a background volume for a single frame

        Args:
            frame_idx (int): 
        """

        frames, _ = self.get_images()
        masks = self.get_masks()

        frames = mask_out_foreground(frames, masks)

        noise = self.get_frame_noise(frame_idx)

        aligned_frames = []
        for i, frame in enumerate(frames):
           
            if i == frame_idx:
                aligned_frames.append(frame)
            else:
                homography = self.calculate_homography_between_two_frames(frame_idx, i)
                aligned_frames.append(cv2.warpPerspective(frame, homography, (self.frame_size[0], self.frame_size[1])))

        aligned_frames = self.add_background_noise(aligned_frames, noise)

        # save results if path is specified
        if self.save_dir is not None:

            for i, frame in enumerate(aligned_frames):
                cv2.imwrite(path.join(self.save_dir, f"{i*self.interval:05}.jpg"), frame)

        return aligned_frames

    def construct_full_volume(self) -> Tuple[list, np.array]:
        """
        construct a background volume:
            - create a global image window in which all frames can fit when aligned
            - remove foreground objects from the frames
            - warp and align all frames using the homographies
            - add a static spatial noise to the empty regions
            - save aligned frames             
        """

        # get frames and frame tensors
        frames, _ = self.get_images()
        
        # get masks
        masks = self.get_masks()

        # mask out the foreground objects
        frames = mask_out_foreground(frames, masks)

        # align all frames
        aligned_frames = self.align_frames(frames)

        # add static noise to empty regions
        aligned_frames = self.add_background_noise(aligned_frames, self.spatial_noise)

        # save results if path is specified
        if self.save_dir is not None:

            for i, frame in enumerate(aligned_frames):
                cv2.imwrite(path.join(self.save_dir, f"{i*self.interval:05}.jpg"), frame)

        return aligned_frames

    def add_background_noise(self, frames: list, noise: np.array) -> list:
        """
        Add a static spatial noise to the empty regions of the warped frames

        Args:
            frames (list[np.array]): list of frames
            noise (np.array): spatial noise, should be of the same dimensions as the frames

        Returns:
            processed_frames (list): list of frames with the random noise added
        """
        processed_frames = []
        for frame in frames:
            assert frame.shape == noise.shape, f"The shape of the image and the noise should be the same. Got {frame.shape} (image) and {noise.shape} (noise)"

            # NOTE fix the threshold
            _, mask = cv2.threshold(frame, 0, 1, cv2.THRESH_BINARY_INV)
            frame = frame + noise * mask
            processed_frames.append(frame)
        
        return processed_frames

    def _initialize_masks(self, n_masks: int, mask_shape: list) -> list:
        """
        Initialize a list of masks with the correct size. All values initialized to zero

        The directory with the relevant masks are passed in order to inform the amount of masks and the 
        size of the masks that will be initialized

        Args:
            n_masks (int): number of masks that need to be initialized
            interval (int): shape of the masks

        Returns:
            masks (list[np.array]): list of np.arrays with all zero values
        """
        return [np.zeros(mask_shape) for _ in range(n_masks)]

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
            warped_corners = cv2.perspectiveTransform(corners, homography)
            all_corners.append(warped_corners)
            corners = warped_corners
        all_corners = np.concatenate(all_corners, axis=0)

        # find the extreme values in the corners
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        return xmin, ymin, xmax, ymax
        
    def align_frames(self, frames: list) -> list:
        """
        Expand, warp and align a set of frames such that they overlay each other correctly 
        using the respective homographies

        Args:
            frames (list[np.array]): list of np.arrays containing the frames
        """
        
        Ht = get_translation_matrix(-self.xmin, -self.ymin)

        warped_frames = []
        for i in range(1, len(frames)):
            
            homography = self.calculate_homography_between_two_frames(0, i)
            warped_frames.append(cv2.warpPerspective(frames[i], Ht @ homography, (self.xmax - self.xmin, self.ymax - self.ymin)))

        source_padded = np.zeros((self.ymax-self.ymin, self.xmax-self.xmin, 3), dtype=np.uint8)
        source_padded[-self.ymin:self.frame_size[1]-self.ymin, -self.xmin:self.frame_size[0]-self.xmin] = frames[0]

        warped_frames.insert(0, source_padded)

        return warped_frames

    def get_frame_noise(self, frame_idx: int) -> np.array:
        """
        Get a perspective projection of the static spatial noise into the given frame

        Args:
            frame_idx (int): index of the relevant frame 
        
        Returns:
            warped_noise (np.array): the spatial noise warped into the selected frames perspective
        """

        homography = self.calculate_homography_between_two_frames(0, frame_idx)
        Ht = get_translation_matrix(-self.xmin, -self.ymin)

        transformation_matrix = np.linalg.inv(Ht @ homography)

        warped_noise = cv2.warpPerspective(self.spatial_noise, transformation_matrix, (self.frame_size[0], self.frame_size[1]))
        return warped_noise

    def draw_frame_border(self, image, Ht, homography):
        w, h = self.frame_size
        rect = np.float32([[0, 0], 
                              [0, h],
                              [w, h], 
                              [w, 0]]).reshape(-1, 1, 2) 
        rect = cv2.perspectiveTransform(rect, Ht@homography)+0.5
        rect = rect.astype(np.int32)
        
        image = cv2.polylines(image, [rect], True, (0,0,255), 8)
        return image
