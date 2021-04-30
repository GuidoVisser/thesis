
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

