import numpy as np
import math

def create_test_path(path_len: int, max_translation: float=10.0, max_rotation: float=2*math.pi) -> np.ndarray:
    """
    Create a dummie 6 DoF camera path for testing

    Args:
        path_len (int): Length of the path in discrete steps
        max_translations (float): max translation in the x, y and z direction
        max_rotation (float): maximum camera rotation
    
    Returns:
        path (np.ndarray): 6 x path_len ndarray. The top three are translations and the bottom three are rotations
    """

    path_ = []
    for _ in range(3):
        path_.append(np.sort(np.random.uniform(low=0., high=max_translation, size=path_len)))

    for _ in range(3):
        path_.append(np.sort(np.random.uniform(low=0., high=max_rotation, size=path_len)))

    return np.stack(path_)

def find_center_of_path(path: np.ndarray, rot_trans_score_ratio: float = 3.) -> int:
    """
    Find the 'center' of a 6 DoF camera path.

    Args:
        path (np.ndarray[6, path_length]): path with dimensions 6 (dof) x length
        rot_trans_score_ratio (float): ratio between movement score assigned to rotation of the camera compared 
                                       to translation of the camera

    Returns:
        center_idx (int): index of steps along path length that is at the center of the path according to the score
    """
    path_length = path.shape[1]
    
    # get array of cumulitative movement cost
    cost = [np.sum(path[:3, 0]) + rot_trans_score_ratio * np.sum(path[3:, 0])]
    for i in range(1, path_length):
        cost.append(cost[i-1] + np.sum(path[:3, i]) + rot_trans_score_ratio * np.sum(path[3:, i]))
    cost = np.array(cost)

    # find and return center of array
    center_idx = np.argmin(np.absolute(cost - cost.mean()))
    return center_idx

path = create_test_path(10)
center = find_center_of_path(path)




