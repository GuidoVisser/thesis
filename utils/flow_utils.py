from math import sqrt
import torch

from models.RAFT.utils.frame_utils import readFlow

def normalize_optical_flow(flow):
    """
    Normalize the optical flow using the image dimensions. We use the length of the image diagonal as the normalizing constant
    to make sure that any following process is equally sensitive to both u and v dimensions.

    Args:
        flow (torch.Tensor[B, 2, W, H]): batch of optical flow estimations between two frames

    Returns:
        flow_norm (torch.Tensor[B, 2, W, H]): batch of the same optical flow but normalized between 0 and 1
    """
    W, H = list(flow.size()[-2:])
    diag = sqrt(W**2 + H**2)

    flow_norm = torch.div(flow, diag)

    return flow_norm

def load_flow_frame(filepath):
    """
    Use readFlow from RAFT implementation

    Args:
        filepath (str): path to .flo file for the frame

    Returns:
        flow (torch.Tensor[1, 2, W, H]): A tensor with the optical flow of the current frame to the next 
    """
    return torch.from_numpy(readFlow(filepath)).permute(2, 0, 1).unsqueeze(0)