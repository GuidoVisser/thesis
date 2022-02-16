import numpy as np

def remove_foreground_features(coords: list, masks: list) -> list:
    """
    Remove the features that belong to a foreground object

    Args:
        coords (list[list[list[np.array]]]): list of coordinate pairs
        masks (list[torch.Tensor]): list of mask pairs

    Rerurns:
        adjusted_coords (list): the same list of coordinate pairs as the input, with the foreground features removed
    """

    assert len(coords) == len(masks) - 1 , f"Should get N-1 coords for N masks. Got {len(coords)} coords and {len(masks)} masks"

    adjusted_coords = []
    for i in range(len(coords)):
        coords0, coords1 = coords[i]
        mask0, mask1 = masks[i], masks[i+1]

        adjusted_coords0, adjusted_coords1 = [], []
        for j in range(len(coords0)):
            if mask0[int(coords0[j][1]), int(coords0[j][0])] == 0 and mask1[int(coords1[j][1]), int(coords1[j][0])] == 0:
                adjusted_coords0.append(coords0[j])
                adjusted_coords1.append(coords1[j])
        adjusted_coords.append([np.array(adjusted_coords0), np.array(adjusted_coords1)])

    return adjusted_coords


def mask_out_foreground(frames: list, masks: list) -> list:
    """
    Remove the masked area's from each frame

    Args:
        frames (list[np.array]): list of frames
        masks (list[np.array]): list of masks

    Args:
        masked_frames (list[np.array]): list of frames with masked areas set to zero.
    """

    masked_frames = []
    for i in range(len(frames)):
        frame = frames[i]
        mask = np.stack([masks[i]]*3, axis=2)

        assert frame.shape[:2] == mask.shape[:2], f"Width and Height of frame and mask should be the same. Got {frame.shape[:2]} (frame) and {mask.shape[:2]} (mask)"

        masked_frames.append(frame * np.logical_not(mask))
    
    return masked_frames