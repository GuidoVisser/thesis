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
                 lambda_stabilization: float = 0.001,
                 lambda_dynamics_reg_corr: float = 0.001,
                 lambda_dynamics_reg_diff: float = 0.001) -> None:
        super().__init__()

        self.criterion = nn.L1Loss()
        self.mask_criterion = MaskLoss()

        self.lambda_alpha_l0          = lambda_alpha_l0
        self.lambda_alpha_l1          = lambda_alpha_l1
        self.lambda_mask_bootstrap    = lambda_mask
        self.lambda_recon_flow        = lambda_recon_flow
        self.lambda_recon_warp        = lambda_recon_warp
        self.lambda_alpha_warp        = lambda_alpha_warp
        self.lambda_stabilization     = lambda_stabilization
        self.lambda_dynamics_reg_corr = lambda_dynamics_reg_corr
        self.lambda_dynamics_reg_diff = lambda_dynamics_reg_diff


    def __call__(self, predictions: dict, targets: dict) -> torch.Tensor:

        # Ground truth values
        flow_gt         = targets["flow"]            # [B, T, 2, H, W]
        rgb_gt          = targets["rgb"]             # [B, T, 3, H, W]
        masks           = targets["masks"]           # [B, T, L, 1, H, W]
        binary_masks    = targets["binary_masks"]    # [B, T, L, 1, H, W]
        flow_confidence = targets["flow_confidence"] # [B, T, 1, H, W]

        ### Main loss

        # Model predictions
        rgba_reconstruction = predictions["rgba_reconstruction"]        # [B, T, 4, H, W]
        flow_reconstruction = predictions["flow_reconstruction"]        # [B, T, 2, H, w]
        rgb_reconstruction  = rgba_reconstruction[:, :, :3]             # [B, T, 3, H, W]
        alpha_composite     = rgba_reconstruction[:, :, 3:]             # [B, T, 1, H, W]
        alpha_layers        = predictions["layers_rgba"][..., 3:, :, :] # [B, T, L, 1, H, W]

        # Calculate main loss
        rgb_reconstruction_loss  = self.calculate_loss(rgb_reconstruction, rgb_gt)
        flow_reconstruction_loss = self.calculate_loss(flow_reconstruction * flow_confidence, flow_gt * flow_confidence)
        mask_bootstrap_loss      = self.calculate_loss(alpha_layers, masks, mask_loss=True)
        alpha_reg_loss           = cal_alpha_reg(alpha_composite * 0.5 + 0.5, self.lambda_alpha_l1, self.lambda_alpha_l0)
        dynamics_reg_loss        = self.cal_dynamics_reg(self.rearrange_t2b(alpha_layers), self.rearrange_t2b(binary_masks))

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
               dynamics_reg_loss + \
               self.lambda_recon_flow     * flow_reconstruction_loss + \
               self.lambda_mask_bootstrap * mask_bootstrap_loss + \
               self.lambda_stabilization  * stabilization_loss

        # create dict of all separate losses for logging
        loss_values = {
            "total":                          loss.item(),
            "rgb_reconstruction_loss":        rgb_reconstruction_loss.item(),
            "alpha_regularization_loss":      alpha_reg_loss.item(),
            "dynamics_regularization_loss":   dynamics_reg_loss.item(),
            "flow_reconstruction_loss":       self.lambda_recon_flow     * flow_reconstruction_loss.item(),
            "mask_bootstrap_loss":            self.lambda_mask_bootstrap * mask_bootstrap_loss.item(),
            "camera_stabilization_loss":      self.lambda_stabilization  * stabilization_loss.item(),
            "brightness_regularization_loss": brightness_regularization_loss.item(),
            "background_offset_loss":         background_offset_loss.item(),
            "lambda_flow_reconstruction":     self.lambda_recon_flow,
            "lambda_mask_bootstrap":          self.lambda_mask_bootstrap,
            "lambda_stabilization":           self.lambda_stabilization,
            "lambda_alpha_l0":                self.lambda_alpha_l0,
            "lambda_alpha_l1":                self.lambda_alpha_l1,
            "lambda_dynamics_reg_diff":       self.lambda_dynamics_reg_diff,
            "lambda_dynamics_reg_corr":       self.lambda_dynamics_reg_corr
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

    def cal_dynamics_reg(self, alpha_layers: torch.Tensor, binary_masks: torch.Tensor) -> torch.Tensor:
        """
        Calculate the dynamics regularization loss that guides the learning to discourage the 
        object alpha layers to learn to assign alpha to dynamic regions of the background and leave the
        dynamic background layer to take care of this.

        Args:
            alpha_layers (torch.Tensor) [B',     L, 1, H, W] (B' = B * 2)
            binary_masks (torch.Tensor) [B', L - 2, 1, H, W]
        """

        dynamics_layer = alpha_layers[:, 1].unsqueeze(1) * .5 + .5  # [B',   1, 1, H, W]
        object_layers  = alpha_layers[:, 2:] * .5 + .5              # [B', L-2, 1, H, W]

        dynamics_layer.expand(object_layers.shape)

        alpha_diff = self.lambda_dynamics_reg_diff * torch.maximum((object_layers - dynamics_layer), torch.zeros_like(dynamics_layer))
        alpha_corr = self.lambda_dynamics_reg_corr * (object_layers * dynamics_layer)

        loss = torch.mean((1 - binary_masks) * (alpha_corr + alpha_diff))

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
