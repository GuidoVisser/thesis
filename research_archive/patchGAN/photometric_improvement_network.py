from glob import glob
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from InputProcessing.patchSampler import PatchSampler
from models.DynamicLayerDecomposition.modules.modules_2d import ConvBlock2D
from torch.nn.parallel import DataParallel

class CriticNet(nn.Module):   
    def __init__(self, conv_channels, patch_size):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock2D(6,                 conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'), # 1/2 
            ConvBlock2D(conv_channels * 2, conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'), # 1/4
            ConvBlock2D(conv_channels * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'), # 1/8
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'), # 1/16
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky')  # 1/32
        )
        self.linear_out = nn.Linear(conv_channels * 4 * (patch_size // 32)**2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        b = x.shape[0]
        
        x = self.layers(x)
        x = x.view(b, -1)

        return torch.mean(self.linear_out(x))


class GeneratorNet(nn.Module):   
    def __init__(self, conv_channels):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock2D(6,                 conv_channels    , ksize=4, stride=1, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels,     conv_channels * 2, ksize=4, stride=1, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 2, conv_channels * 4, ksize=4, stride=1, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 4, conv_channels * 4, ksize=4, stride=1, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 4, conv_channels * 2, ksize=4, stride=1, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels * 2, conv_channels,     ksize=4, stride=1, norm=nn.BatchNorm2d, activation='leaky'),
            ConvBlock2D(conv_channels,                     3, ksize=4, stride=1, norm=nn.BatchNorm2d, activation='tanh'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PatchImprovementGAN(nn.Module):

    def __init__(self,
                 conv_channels: int,
                 lambda_photo: float,
                 critic_clamp: float,
                 patch_size: int, 
                 patch_sampler: PatchSampler,
                 writer: SummaryWriter):
        super().__init__()

        self.lambda_photo = lambda_photo
        self.critic_clamp = critic_clamp

        self.patch_sampler = patch_sampler
        self.critic    = DataParallel(CriticNet(conv_channels=conv_channels, patch_size=patch_size))
        self.generator = DataParallel(GeneratorNet(conv_channels=conv_channels))
        self.writer    = writer

        self.optimizer_critic    = Adam(self.critic.parameters())
        self.optimizer_generator = Adam(self.generator.parameters())

    def update_cycle(self, input: torch.Tensor, ground_truth: torch.Tensor, global_step: int) -> torch.Tensor:
        """
        Perform one full update cycle on both the generator and the critic

        Args:
            input (dict): input for the models
            global_step (int): index for logging
        """
        
        self.critic_step(input, ground_truth, global_step)
        self.generator_step(input, ground_truth, (global_step+1)*self.patch_sampler.patches_per_input)

    def critic_step(self, input: torch.Tensor, ground_truth: torch.Tensor, global_step: int) -> None:
        """
        Perform one update step of the critic network
        
        Args:
            input (torch.Tensor)
            ground_truth (torch.Tensor)
            global_step (int): global step in summary writer
        """

        rgb    = input[:, :,  :3]
        alphas = input[:, :, 3:4]

        B, L, C, H, W = rgb.shape
        with torch.no_grad():
            input_gen = rgb.view(B*L, C, H, W)
            input_gen = torch.cat((input_gen, torch.randn_like(input_gen)), dim=1)
            rgb_improved = self.generator(input_gen).view(B, L, C, H, W)
            alphas = alphas.to(rgb_improved.device)
            ground_truth = ground_truth.to(rgb_improved.device)
            input = input.to(rgb_improved.device)
            input_improved = torch.cat((rgb_improved, alphas), dim=2)
        
        patches = self.patch_sampler(input, input_improved, ground_truth)

        patches_per_input = self.patch_sampler.patches_per_input

        patches_fake = torch.cat((patches["composite_patches"], patches["improved_composite_patches"]), dim=1)
        patches_real = torch.cat((patches["composite_patches"], patches["ground_truth_patches"]), dim=1)

        for idx in range(patches_per_input):

            self.critic.zero_grad()
            loss_fake = self.critic(patches_fake[idx:idx+B])
            loss_real = self.critic(patches_real[idx:idx+B])
            loss = loss_fake - loss_real
            loss.backward(retain_graph=True)
            self.optimizer_critic.step()

            for p in self.critic.parameters():
                p.data.clamp_(-self.critic_clamp, self.critic_clamp)

            logs = {"critic_total": loss.item(),
                    "critic_fake":  loss_fake.item(),
                    "critic_real":  loss_real.item()}

            self.writer.add_scalars("losses", logs, global_step=patches_per_input * global_step + idx)

    def generator_step(self, input: torch.Tensor, ground_truth: torch.Tensor, global_step: int) -> None:
        """
        perform one update step with the generator network

        Args:
            input (torch.Tensor)
            ground_truth (torch.Tensor)
            global_step (int): global step in summary writer
        """

        rgb    = input[:, :,  :3]
        alphas = input[:, :, 3:4]

        B, L, C, H, W = rgb.shape
        input_gen = rgb.view(B*L, C, H, W)
        input_gen = torch.cat((input_gen, torch.randn_like(input_gen)), dim=1)
        rgb_improved = self.generator(input_gen).view(B, L, C, H, W)

        alphas = alphas.to(rgb_improved.device)
        ground_truth = ground_truth.to(rgb_improved.device)
        input = input.to(rgb_improved.device)
        input_improved = torch.cat((rgb_improved, alphas), dim=2)

        patches = self.patch_sampler(input, input_improved, ground_truth)

        improved_composite_patches = patches["improved_composite_patches"]
        composite_patches          = patches["composite_patches"]
        alpha_patches              = patches["alpha_patches"]
        ground_truth_patches       = patches["ground_truth_patches"]

        self.generator.zero_grad()
        loss_critic  = -self.critic(torch.cat((composite_patches, improved_composite_patches), dim=1))
        loss_photo   = nn.L1Loss()(improved_composite_patches, (1-alpha_patches) * ground_truth_patches)
        loss = loss_critic + loss_photo * self.lambda_photo

        loss.backward()
        self.optimizer_generator.step()

        logs = {"generator_total": loss.item(),
                "generator_critic": loss_critic.item(),
                "generator_photo": loss_photo.item()}

        self.writer.add_scalars("losses", logs, global_step=global_step)


    @torch.no_grad()
    def improve(self, input: torch.Tensor):
        """
        improve an rgb tensor with the generator network

        Args:
            input (torch.Tensor)
        """
        return self.generator(input)
