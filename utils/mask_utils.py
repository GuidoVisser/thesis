import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from os import path, listdir

def generate_error_mask(predicted_mask, gt_mask):
    """
    Generate a mask that shows the difference between a predicted mask and the 
    ground truth mask

    The error mask will be 
         0 where the predicted mask and ground truth mask agree, 
         1 where the predicted mask is 1 where it should be 0
        -1 where the predicted mask is 0 where it should be 1

    Args:
        predicted_mask (torch.Tensor[B, 1, W, H]): batch of predicted masks
        gt_mask (torch.Tensor[B, 1, W, H]): batch of ground truth masks

    Returns:
        error_mask (torch.Tensor[B, 1, W, H]): batch of error masks
    """

    return predicted_mask - gt_mask

def save_error_mask(mask: torch.Tensor, save_path: str) -> None:
    """
    Convert an error mask to to an RGB image with blue for false postive and
    red for false negative and save the images
    
    Args:
        mask (torch.Tensor[C, H, W]) : sequence of error mask
        save_path (str): path to which to save the mask
    """
    _, H, W = mask.size()

    img = torch.zeros(3, H, W)
    img[1] = mask[mask > 0]
    img[2] = mask[mask < 0]
    
    save_image(img, save_path)      

def next_mask_based_on_flow(mask, flow):
    """
    Create a new mask from a given mask by propagating the pixel values with the optical flow.

    Args:
        mask (torch.Tensor[B, 1, W, H]): Batch of masks that will be used as input
        flow (torch.Tensor[B, 2, W, H]): Batch of flow estimations

    Returns:
        next_mask torch.Tensor[B, 1, W, H]): Batch of mask predictions
    """
    # get relevant dimensions
    batch_size, _, frame_height, frame_width = mask.size()

    # create mesh grid for indexing
    row_coords, col_coords = torch.meshgrid(torch.arange(frame_height), torch.arange(frame_width))

    # stack coords grid in batch dimension
    row_coords = torch.stack([row_coords]*batch_size, 0)
    col_coords = torch.stack([col_coords]*batch_size, 0)

    # create empty mask 
    next_mask = torch.zeros(mask.size())

    # grid of indices for new pixel positions
    pixel_index_row = row_coords + torch.round(flow[:, 1, :, :])
    pixel_index_col = col_coords + torch.round(flow[:, 0, :, :])

    # convert to integers for indexing
    pixel_index_row = pixel_index_row.int()
    pixel_index_col = pixel_index_col.int()

    for b in range(batch_size):
        for x in range(frame_width):
            for y in range(frame_height):

                # calculate new pixel position
                new_x = pixel_index_col[b, x, y].item()
                new_y = pixel_index_row[b, x, y].item()

                # ignore new positions that are out of bounds
                if new_x < 0 or new_y < 0 or new_x >= frame_width or new_y >= frame_height:
                    continue
                    
                # fill in next_mask with pixel values from the current mask, offset by flow vectors
                next_mask[b, :, new_y, new_x] = mask[b, :, y, x]

    return next_mask

def fill_holes_in_mask(mask):
    """
    Fill small holes in a mask.

    TODO
    A small hole is defined as...

    TODO
    We handle non-binary masks like...

    Args:
        mask (torch.Tensor[B, 1, W, H]): Batch of masks

    Returns:
        filled_masks (torch.Tensor[B, 1, W, H]): Batch of masks with small holes filled in
    """
    return

if __name__ == "__main__":

    mask_pred_path = ""
    mask_gt_path = ""
    mask_error_path = ""

    gt_masks = listdir(mask_gt_path)

    for fname in gt_masks:
        mask_pred = torch.tensor(np.array(Image.open(path.join(mask_pred_path, fname))).astype(np.uint8))
        mask_gt = torch.tensor(np.array(Image.open(path.join(mask_gt_path, fname))).astype(np.uint8))

        error_mask = generate_error_mask(mask_pred, mask_gt)
        save_error_mask(error_mask, path.join(mask_error_path, fname))