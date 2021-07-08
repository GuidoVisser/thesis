import cv2
import numpy as np

def get_matching_coordinates(keypoint_matches: list) -> list:
    """
    Given a set of keypoints in a pair of frames, return two sets of coordinates of the keypoints. 
    Keypoints that have been matched to each other are connected to each other by having the same index in the arrays
    
    Args:
        matches (list[dict]): list of dictionaries containing the pixel locations of keypoints in two disinct frames 
                                as well as the matchings between them

    Returns:
        coords (list[np.array]): list of pixel coordinates in the form
            [
                [
                    coords0 (np.array) : coordinates in the first frame of each pair
                    coords1 (np.array) : coordinates in the second frame of each pair
                ],
                [
                    ...
                ],
                ...
            ]
    """

    coords = []
    for pair in keypoint_matches:
        
        # get keypoints in both frames and matching indices
        keypoints0 = pair["keypoints0"]
        keypoints1 = pair["keypoints1"]
        matches    = pair["matches"]

        # get coordinates of matching pairs
        coords0 = []
        coords1 = []
        for i, point in enumerate(keypoints0):
            if matches[i] > -1:
                coords0.append(point)
                coords1.append(keypoints1[matches[i]])
        
        coords.append([
            np.array(coords0),
            np.array(coords1)
        ])

    return coords


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



