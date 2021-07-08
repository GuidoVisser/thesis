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
