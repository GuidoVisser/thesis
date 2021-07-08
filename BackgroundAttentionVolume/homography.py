import cv2
import numpy as np

def extract_homography(coords: list) -> list:
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
        homographies.append(homography)
        
    return homographies



