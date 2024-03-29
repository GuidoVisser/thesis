import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Protocol, Tuple


class LambdaScheduler(object):
    """
    Scheduler for the lambda values
    """

    def __init__(self, lambda_schedule: list) -> None:
        super().__init__()

        assert len(lambda_schedule) % 2 == 1, "Input needs to be list of odd length"

        self.iteration = 0
        self.value = float(lambda_schedule[0])
        self.schedule = self.create_schedule(lambda_schedule[1:])

    def create_schedule(self, lambda_schedule: list):
        """
        Gets an input in the form of a list if the form

        [update_iteration_1, next_value_1, update_iteration, next_value_2, ....]

        creates a dict in the form
        {
            "update_iteration_1": next_value_1,
            "update_iteration_2: next_value_2,
            ...
        }
        """
        return {str(lambda_schedule[i]):lambda_schedule[i+1] for i in range(0, len(lambda_schedule), 2)}

    def update(self):
        """
        Update the iteration count and set the new lambda value if necessary
        """
        self.iteration += 1
        if str(self.iteration) in self.schedule.keys():
            self.value = float(self.schedule[str(self.iteration)])

class DecompositeLoss(nn.Module):
    """
    Decomposite Loss base class

    Handles loss computations
    """
    def __init__(self,
                 lambda_mask,
                 lambda_recon_flow,
                 lambda_recon_depth,
                 lambda_alpha_l0,
                 lambda_alpha_l1,
                 lambda_stabilization,
                 lambda_dynamics_reg_corr,
                 lambda_dynamics_reg_diff,
                 lambda_dynamics_reg_l0,
                 lambda_dynamics_reg_l1,
                 corr_diff,
                 alpha_reg_layers) -> None:
        super().__init__()

        self.criterion = nn.L1Loss()
        self.mask_criterion = MaskLoss()
        self.corr_diff = corr_diff
        self.alpha_reg_layers = alpha_reg_layers

        self.lambda_alpha_l0          = LambdaScheduler(lambda_alpha_l0)
        self.lambda_alpha_l1          = LambdaScheduler(lambda_alpha_l1)
        self.lambda_mask_bootstrap    = LambdaScheduler(lambda_mask)
        self.lambda_recon_flow        = LambdaScheduler(lambda_recon_flow)
        self.lambda_recon_depth       = LambdaScheduler(lambda_recon_depth)
        self.lambda_stabilization     = LambdaScheduler(lambda_stabilization)
        self.lambda_dynamics_reg_corr = LambdaScheduler(lambda_dynamics_reg_corr)
        self.lambda_dynamics_reg_diff = LambdaScheduler(lambda_dynamics_reg_diff)
        self.lambda_dynamics_reg_l0   = LambdaScheduler(lambda_dynamics_reg_l0)
        self.lambda_dynamics_reg_l1   = LambdaScheduler(lambda_dynamics_reg_l1)

    def __call__(self, predictions: dict, targets: dict) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of the loss module
        
        Args:
            predictions (dict): collection of predictions
            targets (dict): collection of targets

        Returns:
            loss (torch.Tensor): complete loss term
            loss_values (dict): collection of separate loss terms and lambdas for logging
        """
        return NotImplemented

    def calculate_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask_loss: bool = False) -> torch.Tensor:
        """
        Calculate the loss between a prediction and target

        Args:
            prediction (torch.Tensor)
            target (torch.Tensor)
            mask_loss (bool): specifies whether to use the L1 criterion are MaskLoss criterion
        
        Returns the loss term
        """
        
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
        
        Returns the dynamic regularization loss term
        """

        dynamics_layer = alpha_layers[:, 1].unsqueeze(1) * .5 + .5  # [B',   1, 1, H, W]
        object_layers  = alpha_layers[:, 2:] * .5 + .5              # [B', L-2, 1, H, W]

        dynamics_layer.expand(object_layers.shape)


        if self.corr_diff:
            alpha_diff = self.lambda_dynamics_reg_diff.value * torch.maximum((object_layers - dynamics_layer), torch.zeros_like(dynamics_layer))
            alpha_corr = self.lambda_dynamics_reg_corr.value * (object_layers * dynamics_layer)

            loss = torch.mean((1 - binary_masks) * (alpha_corr + alpha_diff))
        else:
            # loss = self.lambda_dynamics_reg_diff.value * object_layers - self.lambda_dynamics_reg_corr.value * (object_layers * dynamics_layer)

            # loss = self.lambda_dynamics_reg_corr.value * ((1 - object_layers) * dynamics_layer + object_layers)
            # loss -= self.lambda_dynamics_reg_diff.value * (dynamics_layer)

            loss = (1 - self.lambda_dynamics_reg_diff.value) * (1 - object_layers) * dynamics_layer + object_layers + self.lambda_dynamics_reg_corr.value * dynamics_layer
            loss = (1 - binary_masks) * loss

            loss = self.lambda_dynamics_reg_l1.value * torch.mean(loss) #+ self.lambda_dynamics_reg_l0.value * torch.mean((torch.sigmoid(loss * 5.0) - 0.5) * 2.0)

        return loss

    def cal_alpha_reg(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Calculate the alpha regularization term.

        Args:
            prediction (tensor): composite of predicted alpha layers
        
        Returns the alpha regularization loss term
        """

        loss = 0.
        if self.lambda_alpha_l1.value > 0:
            loss += self.lambda_alpha_l1.value * torch.mean(prediction)
        if self.lambda_alpha_l0.value > 0:
            # Pseudo L0 loss using a squished sigmoid curve.
            l0_prediction = (torch.sigmoid(prediction * 5.0) - 0.5) * 2.0
            loss += self.lambda_alpha_l0.value * torch.mean(l0_prediction)
        return loss

    def update_lambdas(self):
        return NotImplemented


class DecompositeLoss3D(DecompositeLoss):
    """
    Decomposite Loss to use when using 3D convolutions

    Handles loss computations
    """
    def __init__(self,
                 lambda_mask,
                 lambda_recon_flow,
                 lambda_recon_depth,
                 lambda_alpha_l0,
                 lambda_alpha_l1,
                 lambda_stabilization,
                 lambda_dynamics_reg_corr,
                 lambda_dynamics_reg_diff,
                 lambda_dynamics_reg_l0,
                 lambda_dynamics_reg_l1,
                 corr_diff,
                 alpha_reg_layers) -> None:
        
        super().__init__(lambda_mask,
                         lambda_recon_flow,
                         lambda_recon_depth,
                         lambda_alpha_l0,
                         lambda_alpha_l1,
                         lambda_stabilization,
                         lambda_dynamics_reg_corr,
                         lambda_dynamics_reg_diff,
                         lambda_dynamics_reg_l0,
                         lambda_dynamics_reg_l1,
                         corr_diff,
                         alpha_reg_layers)
        
    def __call__(self, predictions: dict, targets: dict) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of the loss module
        
        Args:
            predictions (dict): collection of predictions
            targets (dict): collection of targets

        Returns:
            loss (torch.Tensor): complete loss term
            loss_values (dict): collection of separate loss terms and lambdas for logging
        """
        # Ground truth values
        flow_gt         = targets["flow"]            # [B,    2, T, H, W]
        rgb_gt          = targets["rgb"]             # [B,    3, T, H, W]
        masks           = targets["masks"]           # [B, L, 1, T, H, W]
        binary_masks    = targets["binary_masks"]    # [B, L, 1, T, H, W]
        flow_confidence = targets["flow_confidence"] # [B,    1, T, H, W]

        ### Main loss

        # Model predictions
        rgba_reconstruction  = predictions["rgba_reconstruction"]   # [B,    4, T, H, W]
        flow_reconstruction  = predictions["flow_reconstruction"]   # [B,    2, T, H, w]
        rgb_reconstruction   = rgba_reconstruction[:, :3]           # [B,    3, T, H, W]
        alpha_composite      = rgba_reconstruction[:, 3:]           # [B,    1, T, H, W]
        alpha_layers         = predictions["layers_rgba"][:, :, 3:] # [B, L, 1, T, H, W]

        # Calculate main loss
        rgb_reconstruction_loss   = self.calculate_loss(rgb_reconstruction, rgb_gt)
        flow_reconstruction_loss  = self.calculate_loss(flow_reconstruction * flow_confidence, flow_gt * flow_confidence)
        mask_bootstrap_loss       = self.calculate_loss(alpha_layers, masks, mask_loss=True)
        dynamics_reg_loss         = self.cal_dynamics_reg(alpha_layers, binary_masks)
        if self.alpha_reg_layers:
            alpha_reg_loss        = self.cal_alpha_reg(alpha_layers * 0.5 + 0.5)
        else:
            alpha_reg_loss        = self.cal_alpha_reg(alpha_composite * 0.5 + 0.5)

        if "depth" in targets.keys():
            depth_gt        = targets["depth"]                                              # [B,    1, T, H, W]
            depth_reconstruction = predictions["depth_reconstruction"]                      # [B,    1, T, H, W]
            depth_reconstruction_loss = self.calculate_loss(depth_reconstruction, depth_gt)
        else:
            depth_reconstruction_loss = torch.zeros((1)).to(rgb_reconstruction_loss.device)

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
               self.lambda_recon_flow.value     * flow_reconstruction_loss + \
               self.lambda_recon_depth.value    * depth_reconstruction_loss + \
               self.lambda_mask_bootstrap.value * mask_bootstrap_loss + \
               self.lambda_stabilization.value  * stabilization_loss

        # create dict of all separate losses for logging
        loss_values = {
            "total":                          loss.item(),
            "rgb_reconstruction_loss":        rgb_reconstruction_loss.item(),
            "alpha_regularization_loss":      alpha_reg_loss.item(),
            "dynamics_regularization_loss":   dynamics_reg_loss.item(),
            "flow_reconstruction_loss":       self.lambda_recon_flow.value     * flow_reconstruction_loss.item(),
            "depth_reconstruction_loss":      self.lambda_recon_depth.value    * depth_reconstruction_loss.item(),
            "mask_bootstrap_loss":            self.lambda_mask_bootstrap.value * mask_bootstrap_loss.item(),
            "camera_stabilization_loss":      self.lambda_stabilization.value  * stabilization_loss.item(),
            "brightness_regularization_loss": brightness_regularization_loss.item(),
            "background_offset_loss":         background_offset_loss.item(),
            "lambda_flow_reconstruction":     self.lambda_recon_flow.value,
            "lambda_depth_reconstruction":    self.lambda_recon_depth.value,
            "lambda_mask_bootstrap":          self.lambda_mask_bootstrap.value,
            "lambda_stabilization":           self.lambda_stabilization.value,
            "lambda_alpha_l0":                self.lambda_alpha_l0.value,
            "lambda_alpha_l1":                self.lambda_alpha_l1.value,
            "lambda_dynamics_reg_diff":       self.lambda_dynamics_reg_diff.value,
            "lambda_dynamics_reg_corr":       self.lambda_dynamics_reg_corr.value
        }

        return loss, loss_values

    def update_lambdas(self):
        self.lambda_alpha_l0.update()
        self.lambda_alpha_l1.update()
        self.lambda_mask_bootstrap.update()
        self.lambda_recon_flow.update()
        self.lambda_recon_depth.update()
        self.lambda_stabilization.update()
        self.lambda_dynamics_reg_corr.update()
        self.lambda_dynamics_reg_diff.update()

class DecompositeLoss2D(DecompositeLoss):
    """
    Decomposite Loss to use when using 3D convolutions

    Handles loss computations
    Adds some loss functions for temporal consistency in the output
    """
    def __init__(self,
                 lambda_mask,
                 lambda_recon_flow,
                 lambda_recon_depth,
                 lambda_recon_warp,
                 lambda_alpha_warp,
                 lambda_alpha_l0,
                 lambda_alpha_l1,
                 lambda_stabilization,
                 lambda_dynamics_reg_corr,
                 lambda_dynamics_reg_diff,
                 lambda_dynamics_reg_l0,
                 lambda_dynamics_reg_l1,
                 corr_diff,
                 alpha_reg_layers) -> None:
        super().__init__(lambda_mask, 
                         lambda_recon_flow, 
                         lambda_recon_depth,
                         lambda_alpha_l0,
                         lambda_alpha_l1,
                         lambda_stabilization,
                         lambda_dynamics_reg_corr,
                         lambda_dynamics_reg_diff,
                         lambda_dynamics_reg_l0,
                         lambda_dynamics_reg_l1,
                         corr_diff,
                         alpha_reg_layers)

        self.lambda_recon_warp = LambdaScheduler(lambda_recon_warp)
        self.lambda_alpha_warp = LambdaScheduler(lambda_alpha_warp)

    def __call__(self, predictions: dict, targets: dict) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of the loss module
        
        Args:
            predictions (dict): collection of predictions
            targets (dict): collection of targets

        Returns:
            loss (torch.Tensor): complete loss term
            loss_values (dict): collection of separate loss terms and lambdas for logging
        """
        # Ground truth values
        flow_gt         = targets["flow"]            # [B, 2, 2, H, W]
        rgb_gt          = targets["rgb"]             # [B, 3, 2, H, W]
        masks           = targets["masks"]           # [B, L, 1, 2, H, W]
        binary_masks    = targets["binary_masks"]    # [B, L, 1, 2, H, W]
        flow_confidence = targets["flow_confidence"] # [B, 1, 2, H, W]

        ### Main loss

        # Model predictions
        rgba_reconstruction  = predictions["rgba_reconstruction"]        # [B, 4, 2, H, W]
        flow_reconstruction  = predictions["flow_reconstruction"]        # [B, 2, 2, H, w]
        rgb_reconstruction   = rgba_reconstruction[:, :3]                # [B, 3, 2, H, W]
        alpha_composite      = rgba_reconstruction[:, 3:]                # [B, 1, 2, H, W]
        alpha_layers         = predictions["layers_rgba"][:, :, 3:]      # [B, L, 1, 2, H, W]

        # Calculate main loss
        rgb_reconstruction_loss   = self.calculate_loss(rgb_reconstruction, rgb_gt)
        flow_reconstruction_loss  = self.calculate_loss(flow_reconstruction * flow_confidence, flow_gt * flow_confidence)
        mask_bootstrap_loss       = self.calculate_loss(alpha_layers, masks, mask_loss=True)
        alpha_reg_loss            = self.cal_alpha_reg(alpha_composite * 0.5 + 0.5)
        dynamics_reg_loss         = self.cal_dynamics_reg(alpha_layers, binary_masks)

        if "depth" in targets.keys():
            depth_gt        = targets["depth"]                                              # [B,    1, T, H, W]
            depth_reconstruction = predictions["depth_reconstruction"]                      # [B,    1, T, H, W]
            depth_reconstruction_loss = self.calculate_loss(depth_reconstruction, depth_gt)
        else:
            depth_reconstruction_loss = torch.zeros((1)).to(rgb_reconstruction_loss.device)

        ### Temporal consistency loss

        # Model predictions
        alpha_layers_warped       = predictions["layers_alpha_warped"]   # [B, L, 1, H, W]
        rgb_reconstruction_warped = predictions["reconstruction_warped"] # [B, 4, 2, H, W]

        # Calculate loss for temporal consistency
        alpha_warp_loss              = self.calculate_loss(alpha_layers_warped, alpha_layers[..., 0, :, :])
        rgb_reconstruction_warp_loss = self.calculate_loss(rgb_reconstruction_warped, rgb_reconstruction[:, :, 0])

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
               self.lambda_recon_flow.value     * flow_reconstruction_loss + \
               self.lambda_recon_depth.value    * depth_reconstruction_loss + \
               self.lambda_mask_bootstrap.value * mask_bootstrap_loss + \
               self.lambda_alpha_warp.value     * alpha_warp_loss + \
               self.lambda_recon_warp.value     * rgb_reconstruction_warp_loss + \
               self.lambda_stabilization.value  * stabilization_loss

        # create dict of all separate losses for logging
        loss_values = {
            "total":                          loss.item(),
            "rgb_reconstruction_loss":        rgb_reconstruction_loss.item(),
            "alpha_regularization_loss":      alpha_reg_loss.item(),
            "dynamics_regularization_loss":   dynamics_reg_loss.item(),
            "flow_reconstruction_loss":       self.lambda_recon_flow.value     * flow_reconstruction_loss.item(),
            "depth_reconstruction_loss":      self.lambda_recon_depth.value    * depth_reconstruction_loss.item(),
            "mask_bootstrap_loss":            self.lambda_mask_bootstrap.value * mask_bootstrap_loss.item(),
            "alpha_warp_loss":                self.lambda_alpha_warp.value     * alpha_warp_loss.item(),
            "rgb_reconstruction_warp_loss":   self.lambda_recon_warp.value     * rgb_reconstruction_warp_loss.item(),
            "camera_stabilization_loss":      self.lambda_stabilization.value  * stabilization_loss.item(),
            "brightness_regularization_loss": brightness_regularization_loss.item(),
            "background_offset_loss":         background_offset_loss.item(),
            "lambda_flow_reconstruction":     self.lambda_recon_flow.value,
            "lambda_depth_reconstruction":    self.lambda_recon_depth.value,
            "lambda_mask_bootstrap":          self.lambda_mask_bootstrap.value,
            "lambda_alpha_warp":              self.lambda_alpha_warp.value,
            "lambda_stabilization":           self.lambda_stabilization.value,
            "lambda_warped_reconstruction":   self.lambda_recon_warp.value,
            "lambda_alpha_l0":                self.lambda_alpha_l0.value,
            "lambda_alpha_l1":                self.lambda_alpha_l1.value,
            "lambda_dynamics_reg_diff":       self.lambda_dynamics_reg_diff.value,
            "lambda_dynamics_reg_corr":       self.lambda_dynamics_reg_corr.value
        }

        return loss, loss_values

    def update_lambdas(self):
        self.lambda_alpha_l0.update()
        self.lambda_alpha_l1.update()
        self.lambda_mask_bootstrap.update()
        self.lambda_recon_flow.update()
        self.lambda_recon_depth.update()
        self.lambda_stabilization.update()
        self.lambda_dynamics_reg_corr.update()
        self.lambda_dynamics_reg_diff.update()
        self.lambda_recon_warp.update()
        self.lambda_alpha_warp.update()

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
