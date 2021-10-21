import torch
import torch.nn as nn
import torch.nn.functional as F


class DecompositeLoss(nn.Module):

    def __init__(self,
                 lambda_mask: float = 50.,
                 lambda_recon_flow: float = 1.,
                 lambda_recon_warp: float = 0.,
                 lambda_alpha_warp: float = 0.005,
                 lambda_alpha_l0: float = 0.005,
                 lambda_alpha_l1: float = 0.01,
                 lambda_stabilization: float = 0.001) -> None:
        super().__init__()

        self.criterion = nn.L1Loss()
        self.mask_criterion = MaskLoss()

        self.lambda_alpha_l0       = lambda_alpha_l0
        self.lambda_alpha_l1       = lambda_alpha_l1
        self.lambda_mask_bootstrap = lambda_mask
        self.lambda_recon_flow     = lambda_recon_flow
        self.lambda_recon_warp     = lambda_recon_warp
        self.lambda_alpha_warp     = lambda_alpha_warp
        self.lambda_stabilization  = lambda_stabilization

        self.bootstrap_threshold = 0.005


    def __call__(self, predictions: dict, targets: dict) -> torch.Tensor:

        # Ground truth values
        flow_gt = targets["flow"]                       # [B, 2, 2, H, W]
        rgb_gt  = targets["rgb"]                        # [B, 2, 3, H, W]
        masks   = targets["masks"]                      # [B, 2, 1, 1, H, W]
        flow_confidence = targets["flow_confidence"]    # [B, 2, 1, H, W]

        ### Main loss

        # Model predictions
        rgba_reconstruction = predictions["rgba_reconstruction"]        # [B, 2, 4, H, W]
        flow_reconstruction = predictions["flow_reconstruction"]        # [B, 2, 2, H, w]
        rgb_reconstruction  = rgba_reconstruction[:, :, :3]             # [B, 2, 3, H, W]
        alpha_composite     = rgba_reconstruction[:, :, 3:]             # [B, 2, 1, H, W]
        alpha_layers        = predictions["layers_rgba"][..., 3:, :, :] # [B, 2, L, 1, H, W]

        # Calculate main loss
        rgb_reconstruction_loss  = self.calculate_loss(rgb_reconstruction, rgb_gt)
        flow_reconstruction_loss = self.calculate_loss(flow_reconstruction * flow_confidence, flow_gt * flow_confidence)
        mask_bootstrap_loss      = self.calculate_loss(alpha_layers, masks, mask_loss=True)
        alpha_reg_loss           = cal_alpha_reg(alpha_composite * 0.5 + 0.5, self.lambda_alpha_l1, self.lambda_alpha_l0)

        # Turn off bootstrap loss when loss reaches threshold
        if self.lambda_mask_bootstrap > 0:
            if mask_bootstrap_loss < self.bootstrap_threshold:
                if self.lambda_mask_bootstrap > 0.5:
                    self.lambda_mask_bootstrap *= 0.1
                else:
                    self.lambda_mask_bootstrap = 0
                print(f"Setting alpha bootstrap lambda to {self.lambda_mask_bootstrap}")

        ### Temporal consistency loss

        # Model predictions
        alpha_layers_warped       = predictions["layers_alpha_warped"]   # [B, L, 1, H, W]
        rgb_reconstruction_warped = predictions["reconstruction_warped"] # [B, 2, 4, H, W]

        # Calculate loss for temporal consistency
        alpha_warp_loss              = self.calculate_loss(alpha_layers_warped, alpha_layers[:, 0], t2b=False)
        rgb_reconstruction_warp_loss = self.calculate_loss(rgb_reconstruction_warped, rgb_reconstruction[:, 0], t2b=False)

        ### Adjust for camera stabilization errors

        # Model predictions
        brightness_scale  = predictions["brightness_scale"]
        background_offset = predictions["background_offset"]

        # Calculate loss for camera adjustment
        brightness_regularization_loss = self.calculate_loss(brightness_scale, torch.ones_like(brightness_scale))
        background_offset_loss         = background_offset.abs().mean()

        stabilization_loss = brightness_regularization_loss + background_offset_loss

        # Combine loss values
        loss = rgb_reconstruction_loss + \
               alpha_reg_loss + \
               self.lambda_recon_flow     * flow_reconstruction_loss + \
               self.lambda_mask_bootstrap * mask_bootstrap_loss + \
               self.lambda_alpha_warp     * alpha_warp_loss + \
               self.lambda_recon_warp     * rgb_reconstruction_warp_loss + \
               self.lambda_stabilization  * stabilization_loss

        # create dict of all separate losses for logging
        loss_values = {
            "total":                          loss.item(),
            "rgb_reconstruction_loss":        rgb_reconstruction_loss.item(),
            "alpha_regularization_loss":      alpha_reg_loss.item(),
            "flow_reconstruction_loss":       flow_reconstruction_loss.item(),
            "mask_bootstrap_loss":            mask_bootstrap_loss.item(),
            "alpha_warp_loss":                alpha_warp_loss.item(),
            "rgb_reconstruction_warp_loss":   rgb_reconstruction_warp_loss.item(),
            "camera_stabilization_loss":      stabilization_loss.item(),
            "brightness_regularization_loss": brightness_regularization_loss.item(),
            "background_offset_loss":         background_offset_loss.item(),
            "lambda_flow_reconstruction":     self.lambda_recon_flow,
            "lambda_mask_bootstrap":          self.lambda_mask_bootstrap,
            "lambda_alpha_warp":              self.lambda_alpha_warp,
            "lambda_stabilization":           self.lambda_stabilization,
            "lambda_warped_reconstruction":   self.lambda_recon_warp,
            "lambda_alpha_l0":                self.lambda_alpha_l0,
            "lambda_alpha_l1":                self.lambda_alpha_l1
        }

        return loss, loss_values

    def rearrange_t2b(self, tensor):
        """
        Rearrange a tensor such that the time dimension is stacked in the batch dimension

        [B, 2, ...] -> [B*2, ...]
        """
        assert tensor.size(1) == 2
        return torch.cat((tensor[:, 0], tensor[:, 1]))

    def calculate_loss(self, prediction, target, t2b=True, mask_loss=False):
        
        # Rearrange time dimension into batch dimension
        if t2b:
            prediction  = self.rearrange_t2b(prediction)
            target      = self.rearrange_t2b(target)
        
        # Calculate loss value
        if mask_loss:
            loss = self.mask_criterion(prediction, target)
        else:
            loss = self.criterion(prediction, target)
        
        return loss


##############################################################################################################
# Adapted from Omnimatte: https://github.com/erikalu/omnimatte/tree/018e56a64f389075e548966e4547fcc404e98986 #
##############################################################################################################

class MaskLoss(nn.Module):
    """Define the loss which encourages the predicted alpha matte to match the mask (trimap)."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='none')

    def __call__(self, prediction, target):
        """Calculate loss given predicted alpha matte and trimap.

        Balance positive and negative regions. Exclude 'unknown' region from loss.

        Parameters:
            prediction (tensor) - - predicted alpha
            target (tensor) - - trimap

        Returns: the computed loss
        """
        mask_err = self.loss(prediction, target)
        pos_mask = F.relu(target)
        neg_mask = F.relu(-target)
        pos_mask_loss = (pos_mask * mask_err).sum() / (1 + pos_mask.sum())
        neg_mask_loss = (neg_mask * mask_err).sum() / (1 + neg_mask.sum())
        loss = .5 * (pos_mask_loss + neg_mask_loss)
        return loss


def cal_alpha_reg(prediction, lambda_alpha_l1, lambda_alpha_l0):
    """Calculate the alpha regularization term.

    Parameters:
        prediction (tensor) - - composite of predicted alpha layers
        lambda_alpha_l1 (float) - - weight for the L1 regularization term
        lambda_alpha_l0 (float) - - weight for the L0 regularization term
    Returns the alpha regularization loss term
    """

    loss = 0.
    if lambda_alpha_l1 > 0:
        loss += lambda_alpha_l1 * torch.mean(prediction)
    if lambda_alpha_l0 > 0:
        # Pseudo L0 loss using a squished sigmoid curve.
        l0_prediction = (torch.sigmoid(prediction * 5.0) - 0.5) * 2.0
        loss += lambda_alpha_l0 * torch.mean(l0_prediction)
    return loss