import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from typing import Tuple
from itertools import product
from collections import Counter
from random import uniform, choices, choice, shuffle, randint, random
from os import path, listdir

from models.DynamicLayerDecomposition.utils import composite_rgba

class PatchSampler(object):
    """
    Patch Sampler object

    returns a set of patches sampled from either the background layer of the decomposited video or the compostite 
    of the foreground and background layers.
    Furthermore a set of ground truth patches is returned. Per input patch the ground truth patch of that location 
    in the image as well as a random ground truth patch is given.
    Finally a set of alpha layer patches is returned to be used to weigh down the photometric loss
    """

    def __init__(self, 
                 include_fg_rate: float, 
                 patches_per_input: int,
                 patch_size: tuple,
                 frame_size: tuple,
                 img_root: str) -> None:
        """
        Create a PatchSampler object

        Args:
            include_fg_rate (float): rate at which the foreground layer is included in the patches
            patches_per_image (int): number of patches that is sampled for each image
            patch_size (tuple[int, int]): height and width of the patches in pixels
            frame_size (tuple[int, int]): height and width of the frames in pixels
            root_dir (str): root directory for the input
        """
        super().__init__()
        self.include_fg_rate = include_fg_rate
        self.patches_per_input = patches_per_input
        self.patch_size = patch_size
        self.frame_size = frame_size
        self.x_ratio = patch_size[0] / frame_size[0]
        self.y_ratio = patch_size[1] / frame_size[1]

        self.img_paths = [path.join(img_root, fn) for fn in sorted(listdir(img_root))]

    def __call__(self, rgba_layers: torch.Tensor, rgba_layers_improved: torch.Tensor, ground_truth: torch.Tensor) -> dict:
        """
        Forward pass of the module
        """
        
        # B, L, C, T, H, W = rgba_layers.shape

        # rgba_layers_  = rgba_layers.permute(0, 3, 1, 2, 4, 5).view(B*T, L, C, H, W)
        # ground_truth_ = ground_truth.permute(0, 2, 1, 3, 4).view(B*T, 3, H, W) 

        batch_size = rgba_layers.shape[0]

        # Get target patches
        target_patches = self._get_target_patches(batch_size)

        # Get patches of composites and ground truth
        improved_composite_patches, composite_patches, alpha_patches, ground_truth_patches = self._get_rgba_patches(rgba_layers_improved, rgba_layers, ground_truth)

        out = {
            "improved_composite_patches": improved_composite_patches,
            "composite_patches": composite_patches,
            "alpha_patches": alpha_patches,
            "ground_truth_patches": ground_truth_patches,
            "target_patches": target_patches
        }

        return out

    def _get_target_patches(self, batch_size: int) -> torch.Tensor:
        """
        Get a list of target patches from the image directory. The patches are random selections from random images
        
        Args:
            batch_size (int): batch size of input

        TODO support for time dimension
        """
        # Get a dictionary of filepaths with a random choice of how many times will be sampled from that image
        targets = choices(self.img_paths, k=self.patches_per_input*batch_size)
        targets = dict(Counter(targets).items())

        # load images and sample the specified amount of patches
        target_patches = []
        for (target_path, target_count) in targets.items():

            # load image
            target = torch.from_numpy(cv2.resize(np.float32(cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)), self.frame_size)).permute(2, 0, 1) / 255.
            target = target * 2 - 1

            # sample patches
            for _ in range(target_count):
                target_patches.append(self._get_random_patches(target.unsqueeze(0))[0])
        
        # shuffle list
        shuffle(target_patches)

        target_patches = torch.cat(target_patches)

        return target_patches


    def _get_rgba_patches(self, rgba_layers_improved: torch.Tensor, rgba_layers: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a set of random patches from the rgba layers with random permutations of compositing and corresponding ground truth patches

        Args:
            rgba_layers (torch.Tensor)
            ground_truth (torch.Tensor)
        """

        composites_rgb, composites_alpha = self._get_composites(rgba_layers)
        impr_composites_rgb, _ = self._get_composites(rgba_layers_improved)
        impr_composite_patches, composite_patches, alpha_patches, ground_truth_patches = [], [], [], []

        for _ in range(self.patches_per_input):
        
            if random() < self.include_fg_rate:
                random_choice = 0
            else:
                random_choice = randint(1, len(composites_rgb)-1)

            impr_composite_patch, composite_patch, alpha_patch, ground_truth_patch = self._get_random_patches(impr_composites_rgb[random_choice], 
                                                                                                                composites_rgb[random_choice], 
                                                                                                                composites_alpha[random_choice], 
                                                                                                                ground_truth)
            
            impr_composite_patches.append(impr_composite_patch)
            composite_patches.append(composite_patch)
            alpha_patches.append(alpha_patch)
            ground_truth_patches.append(ground_truth_patch)

        composite_patches      = torch.cat(composite_patches)
        alpha_patches          = torch.cat(alpha_patches)
        ground_truth_patches   = torch.cat(ground_truth_patches)
        impr_composite_patches = torch.cat(impr_composite_patches)

        return impr_composite_patches, composite_patches, alpha_patches, ground_truth_patches

    def _get_random_patches(self, *inputs: list) -> list:
        """
        return a random patch of the given input tensors. The same spatial sampling will be done on all given tensors

        Args:
            input (list[torch.Tensor]): list of tensors that ought to be sampled from
        """
        batch_size = inputs[0].shape[0]

        sample_grid = []
        for _ in range(batch_size):
            # Get the position of the bottom left corner of the patch patch
            x_pos = uniform(-1, 1 - self.x_ratio * 2)
            y_pos = uniform(-1, 1 - self.y_ratio * 2)

            # create a sampling grid at the given position
            u = torch.linspace(x_pos, x_pos + self.x_ratio * 2, self.patch_size[0]).unsqueeze(0).repeat(self.patch_size[1], 1)
            v = torch.linspace(y_pos, y_pos + self.y_ratio * 2, self.patch_size[1]).unsqueeze(1).repeat(1, self.patch_size[0])
            sample_grid.append(torch.stack((u, v), dim=-1))
        sample_grid = torch.stack(sample_grid)

        # sample a patch from the input
        outputs = []
        for input in inputs:
            outputs.append(F.grid_sample(input, sample_grid.to(input.device), mode='bilinear'))

        return outputs

    def _get_composites(self, rgba):
        """
        Get a list of composites that include all possible perturbations of included layers
        
        Args:
            rgba (torch.Tensor): rgba layers
        """

        permutations = list(product([False, True], repeat=rgba.shape[1] - 1))

        composites_rgb = []
        composites_alpha = []

        for permutation in permutations:
            composite = rgba[:, 0]

            for layer in range(1, rgba.shape[1]):

                if permutation[layer - 1]:
                    composite = composite_rgba(composite, rgba[:, layer])

            composites_rgb.append(composite[:, :3])
            composites_alpha.append(composite[:, 3:4])

        return composites_rgb, composites_alpha


    def demo_out(self, out: dict, idx: int = 0) -> None:
        """
        Show a visual demo of the output patches

        Args:
            out (dict): dictionary containing the output of the module
            idx (int): index in batch dimension which patch to show
        """

        composite    = out["composite_patches"][idx].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        alpha        = out["alpha_patches"][idx].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        ground_truth = out["ground_truth_patches"][idx].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
        target       = out["target_patches"][idx].detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5

        cv2.imshow("composite", composite)
        cv2.imshow("alpha", alpha)
        cv2.imshow("ground_truth", ground_truth)
        cv2.imshow("target", target)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



