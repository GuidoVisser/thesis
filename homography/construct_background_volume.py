from typing import Tuple, Union
from homography.feature_matching import get_model_config, extract_and_match_features
from homography.extract_homography import get_corresponding_coordinates, extract_homography
from homography.align_frames import align_frames
from homography.remove_foreground import remove_foreground_features, mask_out_foreground

from models.SuperGlue.models.utils import process_resize, frame2tensor

import cv2
from os import listdir, path
from utils.utils import create_dir
import numpy as np
from typing import Union

def initialize_masks(n_masks: int, mask_shape: list) -> list:
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


def get_masks(mask_dirs: Union[str, list],
              resize: list = [864, 480], # STM uses 864 x 480
              interval: int = 1,
              binary_threshold: float = 0.7) -> list:
    """
    Load all object masks of the frames that are being considered and construct a single global
    mask for all foreground objects 

    Args:
        mask_dir (str or list[str]): path(s) to the directory where the masks are kept
        interval (int): interval between frames
        binary_threshold (float): threshold for a pixel being masked by a non-binary mask
    
    Returns:
        masks (list): list of combined masks
    """

    if isinstance(mask_dirs, str):
        mask_dirs = [mask_dirs]

    n_masks = len(listdir(mask_dirs[0]))//interval
    masks = initialize_masks(n_masks, (resize[1], resize[0]))
    for dir in mask_dirs:

        mask_paths = [path.join(dir, mask) for mask in sorted(listdir(dir))]
        for i, mask_path in enumerate(mask_paths):

            # skip iteration if it is not a valid iteration according to the interval
            if i % interval != 0:
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # resize mask
            w, h = mask.shape[1], mask.shape[0]
            w_new, h_new = process_resize(w, h, resize)
            mask = cv2.resize(mask, (w_new, h_new))
            
            # ensure that mask is binary
            _, mask = cv2.threshold(mask, binary_threshold, 1, cv2.THRESH_BINARY)

            masks[i//interval] = np.add(masks[i//interval], mask)

    for mask in masks:
        np.minimum(mask, np.ones(mask.shape), mask)

    return masks
        



def get_images(img_dir: str, 
               device: str,
               resize: list = [864, 480], # STM uses 864 x 480
               interval: int = 1) -> list:
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

    # construct the full paths to the frames
    img_paths = [path.join(img_dir, frame) for frame in sorted(listdir(img_dir))]

    # loop through file paths and process images
    frames, frame_tensors = [], []
    for i, frame_path in enumerate(img_paths):

        if i % interval != 0:
            continue

        # load frame
        frame = cv2.imread(frame_path)

        # resize image
        w, h = frame.shape[1], frame.shape[0]
        w_new, h_new = process_resize(w, h, resize)
        frame = cv2.resize(frame, (w_new, h_new))
        
        # convert to grayscale for SuperGlue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float32')

        # add image and image tensor to output
        frames.append(frame)
        frame_tensors.append(frame2tensor(frame_gray, device))

    return frames, frame_tensors

def add_background_noise(frames: list, low: float = 0., high: float = 255.) -> list:
    """
    Add a static spatial noise to the empty regions of the warped frames

    Args:
        frames (list[np.array]): list of frames
        low (float): lower bound for the random noise
        high (float): upper bound for the random noise

    Returns:
        processed_frames (list): list of frames with the random noise added
    """
    noise = np.random.uniform(low=low, high=high, size=frames[0].shape)

    processed_frames = []
    for frame in frames:
        _, mask = cv2.threshold(frame, 0, 1, cv2.THRESH_BINARY_INV)
        frame = frame + noise * mask
        processed_frames.append(frame)
    
    return processed_frames


def construct_volume(img_path: str, 
                     mask_path: str,
                     device: str, 
                     resize: list = [864, 480], # STM uses 864 x 480
                     frame_interval: int=1, 
                     save_dir: Union[str, None] = None) -> Tuple[list, np.array]:
    """
    construct a background volume:
        - find feature matches
        - estimate homographies between frames
        - create a global image window in which all frames can fit when aligned
        - warp and align all frames using the homographies
        - save aligned frames 

    Args:
        
    """

    # get frames and frame tensors
    frames, frame_tensors = get_images(img_path, device, resize, frame_interval)
    
    # get masks
    masks = get_masks(mask_path, resize, frame_interval)

    # extract features using SuperPoint and match the features using SuperGlue
    feature_matches = extract_and_match_features(frame_tensors, device, get_model_config())
    
    # get the coordinates of matching features for RANSAC
    coords = get_corresponding_coordinates(feature_matches)

    # remove foreground features for cleaner homography estimation
    coords = remove_foreground_features(coords, masks)
    
    # apply RANSAC and get homographies from coordinates 
    homographies = extract_homography(coords)

    # mask out the foreground objects
    frames = mask_out_foreground(frames, masks)

    # align all frames
    aligned_frames = align_frames(frames, homographies)

    # add static noise to empty regions
    aligned_frames = add_background_noise(aligned_frames)

    # save results if path is specified
    if save_dir is not None:

        # make sure save_dir exists
        create_dir(save_dir)

        for i, frame in enumerate(aligned_frames):
            cv2.imwrite(path.join(save_dir, f"{i*frame_interval:05}.jpg"), frame)

    return aligned_frames
