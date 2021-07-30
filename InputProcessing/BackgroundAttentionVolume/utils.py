import numpy as np

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

def get_scale_matrix(source_w, source_h, target_w, target_h):

    x_scale = float(target_w) / float(source_w)
    y_scale = float(target_h) / float(source_h)

    return np.array([[x_scale, 0, 0],
                     [0, y_scale, 0],
                     [0,    0,    1]])