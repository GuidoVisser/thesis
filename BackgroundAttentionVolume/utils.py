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