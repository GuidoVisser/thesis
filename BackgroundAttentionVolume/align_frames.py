from typing import Tuple
import cv2
import numpy as np
from numpy.core.numeric import outer


def align_two_frames(source_frame: np.array, target_frame: np.array, homography: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    Warp a frame using a homography matrix

    Args:
        source_frame (np.array): input frame that needs to be warped
        target_frame (np.array): target frame that the source frame must align with
        homography (np.array): homography matrix

    Returns:
        warped_frame (np.array): warped source frame expanded and translated such that it aligns with the input frame
        padded_source (np.array): source frame padded to match the dimensions of the warped source frame
        combined (np.array): the warped source frame with the source frame overlayed on it
    """
    hs, ws = source_frame.shape[:2]
    source_corners = np.float32([[0, 0], 
                                 [0, hs],
                                 [ws,hs], 
                                 [ws, 0]]).reshape(-1, 1, 2)

    ht, wt = target_frame.shape[:2]
    target_corners = np.float32([[0, 0],
                                [0, ht],
                                [wt,ht],
                                [wt, 0]]).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(source_corners, homography)

    all_corners = np.concatenate((warped_corners, target_corners), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    warped_frame = cv2.warpPerspective(source_frame, Ht.dot(homography), (xmax-xmin, ymax-ymin))

    source_padded = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=np.uint8)
    source_padded[t[1]:ht+t[1], t[0]:wt+t[0]] = target_frame
    
    combined = warped_frame.copy()
    combined[t[1]:ht+t[1], t[0]:wt+t[0]] = target_frame

    # draw corners on image
    for corner in warped_corners:
        xc, yc = np.round(corner[0])
        xc = int(xc)
        yc = int(yc)
        cv2.circle(warped_frame, (xc-xmin, yc-ymin), 3, (0,255,0))
        cv2.circle(combined, (xc-xmin, yc-ymin), 3, (255,0,0))

    return warped_frame, source_padded, combined


def get_frame_corners(corners: np.array, homographies: list) -> np.array:
    """
    Get the borders of a video with warped frames such that they overlap

    Args:
        frames (list[np.array]): list of frames in the video
        homographies (list[np.array]): list of homorgraphy projections between the frames

    Returns:
        borders (np.array): coordinates of the outer corners in all warped frames
    """
    all_corners = [corners]
    for homography in homographies:
        warped_corners = cv2.perspectiveTransform(corners, homography)
        all_corners.append(warped_corners)
        corners = warped_corners

    return all_corners

def get_outer_corners(corners: list) -> Tuple[float, float, float, float]:
    """
    Get the outer corners from a set of corners

    Args:
        corners (list[np.array]): list of np.arrays containing the corners from a set of warped images

    Returns:
        xmin (float)
        ymin (float)
        xmax (float)
        ymax (float)
    """
    corners = np.concatenate(corners, axis=0)

    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    return xmin, ymin, xmax, ymax

def get_translation_matrix(dx: float, dy: float) -> np.array:
    """
    Construct a translation matrix for a warped frame such that it is moved to the correct 
    position in the overally image

    Args:
        dx (float): translation in the x direction
        dy (float): translation in the y direction

    Returns:
        translation_matrix (np.array)
    """
    return np.array([[1, 0, dx],
                     [0, 1, dy],
                     [0, 0,  1]])

def sequential_homography(homography_sequence: list) -> np.array:
    """
    combine a sequence of homography transformations into one single matrix

    Args:
        homography_sequence (list[np.array]): list of 3x3 homography matrices that will be applied in sequence

    Returns:
        homography (np.array)
    """
    homography = homography_sequence[0]
    for i in range(1, len(homography_sequence)):
        homography = homography_sequence[i] @ homography

    return homography 
    

def align_frames(frames: list, homographies: list, save_results: bool=False) -> list:
    """
    Expand, warp and align a set of frames such that they overlay each other correctly 
    using the respective homographies

    Args:
        frames (list[np.array]): list of np.arrays containing the frames
        homographies (list[np.array]): list of homography matrices that warp each frame i into frame i-1
        save_results (bool, default=False): specifies whether to save the resulting image frames
    """
    
    h, w = frames[0].shape[:2]
    base_corners = np.float32([[0, 0], 
                               [0, h],
                               [w, h], 
                               [w, 0]]).reshape(-1, 1, 2)


    frame_corners = get_frame_corners(base_corners, homographies)
    xmin, ymin, xmax, ymax = get_outer_corners(frame_corners)  
    Ht = get_translation_matrix(-xmin, -ymin)

    warped_frames = []
    for i in range(1, len(frames)):
        
        homography = sequential_homography(homographies[:i])
        warped_frames.append(cv2.warpPerspective(frames[i], Ht @ homography, (xmax-xmin, ymax-ymin)))

    source_padded = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=np.uint8)
    source_padded[-ymin:h-ymin, -xmin:w-xmin] = frames[0]

    warped_frames.insert(0, source_padded)

    return warped_frames