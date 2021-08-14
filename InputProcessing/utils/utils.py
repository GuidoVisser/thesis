import numpy as np
import torch

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

def homogeneous_2d_transform(x, y, m):
    """Applies 2d homogeneous transformation."""
    A = torch.matmul(m, torch.stack([x, y, torch.ones(len(x))]))
    xt = A[0, :] / A[2, :]
    yt = A[1, :] / A[2, :]
    return xt, yt