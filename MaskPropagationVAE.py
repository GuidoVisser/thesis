import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import e, log2, prod
from scipy.stats import norm
from torchvision.utils import make_grid, save_image

class SeqDebug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(torch.sum(x))
        return x


class Conv2dWithSkip(nn.Module):
    def __init__(self, conv_layer, stack=1):
        super().__init__()
        self.conv_layer = conv_layer
        self.stack = stack
    
    def forward(self, x):
        return self.conv_layer(x) + x.repeat([1, self.stack, 1, 1])


class ResidualOut(nn.Module):
    def __init__(self, residuals):
        super().__init__()
        self.residuals = residuals
    
    def forward(self, x):
        self.residuals.append(torch.clone(x))
        return x


class ResidualIn(nn.Module):
    def __init__(self, residuals):
        super().__init__()
        self.residuals = residuals
    
    def forward(self, x):
        return x + self.residuals.pop()


class MaskEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, 
                       num_filters: int = 32,
                       z_dim: int = 20):
        """
        Encoder network for the mask input

        Args:
            num_input_channels (int): number of input channels for the input images. since we're dealing with masks it is 1
            num_filters (int): Number of channels used in the first convolutional layers. Deeper layers will be influenced as well
            z_dim (int): dimensionality of the latent space
        """
        super().__init__()

        self.pad_1 = nn.ZeroPad2d((1, 1, 0, 0))
        self.pad_2 = nn.ZeroPad2d(1)
        self.num_filters = num_filters


        self.conv1a = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.pad_1 = nn.ZeroPad2d((1, 1, 0, 0))

        self.conv2a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.pad_2 = nn.ZeroPad2d(1)

        self.conv3a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.activation = nn.LeakyReLU()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112*num_filters, z_dim)
        )

        self.residuals = []

    def forward(self, x):
        """
        Inputs:
            x (torch.Tensor): Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            x (torch.Tensor): Tensor of shape [B,z_dim]
        """
        # in_shape = tuple(x.size())
        # print(f"Mask  forward: Shape = {in_shape}\r", end='')
        
        x = self.activation(self.conv1a(x) + x.repeat(1, self.num_filters, 1, 1))
        x = self.activation(self.conv1b(x) + x)
        x = self.activation(self.conv1c(x) + x)
        self.residuals.append(torch.clone(x))
        x = self.pool(x)
        x = self.pad_1(x)
        
        x = self.activation(self.conv2a(x) + x)
        x = self.activation(self.conv2b(x) + x)
        x = self.activation(self.conv2c(x) + x)
        self.residuals.append(torch.clone(x))
        x = self.pool(x)
        x = self.pad_2(x)
        
        x = self.activation(self.conv3a(x) + x)
        x = self.activation(self.conv3b(x) + x)
        x = self.activation(self.conv3c(x) + x)
        x = self.pool(x)
        
        x = self.linear(x)
        # print(f"Mask forward : Shape = {in_shape} ==> {tuple(x.size())}")

        return x 


class FlowEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 2, 
                       num_filters: int = 32,
                       z_dim: int = 20):
        """
        Encoder network for the mask input

        Args:
            num_input_channels (int): number of input channels for the input images. since we're dealing with optical flow it's 2
            num_filters (int): Number of channels used in the first convolutional layers. Deeper layers will be influenced as well
            z_dim (int): dimensionality of the latent space
        """
        super().__init__()

        self.pad_1 = nn.ZeroPad2d((1, 1, 0, 0))
        self.pad_2 = nn.ZeroPad2d(1)
        self.num_filters = num_filters

        self.conv1a = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.pad_1 = nn.ZeroPad2d((1, 1, 0, 0))

        self.conv2a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.pad_2 = nn.ZeroPad2d(1)

        self.conv3a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.activation = nn.LeakyReLU()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112*num_filters, z_dim)
        )

    def forward(self, x):
        """
        Inputs:
            x (torch.Tensor): Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            x (torch.Tensor): Tensor of shape [B,z_dim]
        """
        # in_shape = tuple(x.size())
        # print(f"Flow forward : Shape = {in_shape}\r", end='')

        x = self.activation(self.conv1a(x))
        x = self.activation(self.conv1b(x) + x)
        x = self.activation(self.conv1c(x) + x)
        x = self.pool(x)
        x = self.pad_1(x)

        x = self.activation(self.conv2a(x) + x)
        x = self.activation(self.conv2b(x) + x)
        x = self.activation(self.conv2c(x) + x)
        x = self.pool(x)
        x = self.pad_2(x)

        x = self.activation(self.conv3a(x) + x)
        x = self.activation(self.conv3b(x) + x)
        x = self.activation(self.conv3c(x) + x)
        x = self.pool(x)
        x = self.linear(x)

        # print(f"Flow forward : Shape = {in_shape} ==> {tuple(x.size())}")

        return x 

class FrameEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 3, 
                       num_filters: int = 32,
                       z_dim: int = 20):
        """
        Encoder network for the mask input

        Args:
            num_input_channels (int): number of input channels for the input images. since we're 
                                    dealing with optical flow it's 2
            num_filters (int): Number of channels used in the first convolutional layers. Deeper layers 
                                    will be influenced as well
            z_dim (int): dimensionality of the latent space
        """
        super().__init__()

        self.pad_1 = nn.ZeroPad2d((1, 1, 0, 0))
        self.pad_2 = nn.ZeroPad2d(1)
        self.num_filters = num_filters

        self.conv1a = nn.Conv2d(num_input_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.pad_1 = nn.ZeroPad2d((1, 1, 0, 0))

        self.conv2a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.pad_2 = nn.ZeroPad2d(1)

        self.conv3a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.activation = nn.LeakyReLU()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112*num_filters, z_dim)
        )

    def forward(self, x):
        """
        Inputs:
            x (torch.Tensor): Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            z (torch.Tensor): Tensor of shape [B,z_dim]
        """

        # in_shape = tuple(x.size())
        # print(f"Frame forward: Shape = {in_shape}\r", end='')

        x = self.activation(self.conv1a(x))
        x = self.activation(self.conv1b(x) + x)
        x = self.activation(self.conv1c(x) + x)
        x = self.pool(x)
        x = self.pad_1(x)

        x = self.activation(self.conv2a(x) + x)
        x = self.activation(self.conv2b(x) + x)
        x = self.activation(self.conv2c(x) + x)
        x = self.pool(x)
        x = self.pad_2(x)

        x = self.activation(self.conv3a(x) + x)
        x = self.activation(self.conv3b(x) + x)
        x = self.activation(self.conv3c(x) + x)
        x = self.pool(x)
        x = self.linear(x)

        # print(f"Frame forward: Shape = {in_shape} ==> {tuple(x.size())}")

        return x 

class Encoder(nn.Module):
    def __init__(self, num_filters: int = 32, z_dim: int = 20):
        """
        Encoder model that generates a latent vector conditioned on a mask input, a flow input 
        and an image input

        Inputs:
            mask_encoder (torch.nn.Module): encoder model for the mask input
            flow_encoder (torch.nn.Module): encoder model for the flow input
            frame_encoder (torch.nn.Module): encoder model for the frame input
            z_dim (int): dimensionality of the latent space
        """
        super().__init__()

        self.mask_encoder  =  MaskEncoder(num_filters=num_filters, z_dim=z_dim)
        self.residuals     =  self.mask_encoder.residuals
        self.flow_encoder  =  FlowEncoder(num_filters=num_filters, z_dim=z_dim)
        self.frame_encoder = FrameEncoder(num_filters=num_filters, z_dim=z_dim)

        self.mean_out    = nn.Linear(3*z_dim, z_dim)
        self.log_std_out = nn.Linear(3*z_dim, z_dim)
        
        self.z_dim = z_dim

    def forward(self, mask, flow, frame):
        """
        Forward pass of the encoder model. Runs each input through their respective encoder model 
        and combines the results in one latent vector

        Inputs:
            mask  (torch.Tensor[B, 1, 480, 854]): The binary mask defined on the current frame
            flow  (torch.Tensor[B, 2, 480, 854]): The optical flow between the current frame and the next
            frame (torch.Tensor[B, 3, 480, 854]): The image data of the current frame

        Outputs:
            mean    (torch.Tensor[B, z_dim]): A row vector of diminsions {z_dim} for the mean of the latent 
                                                distribution
            log_std (torch.Tensor[B, z_dim]): A row vector of diminsions {z_dim} for the log standard deviation
                                                of the latent distribution
        """

        mask_activation  = self.mask_encoder.forward(mask)
        flow_activation  = self.flow_encoder.forward(flow)
        frame_activation = self.frame_encoder.forward(frame)

        activation = torch.cat((mask_activation, flow_activation, frame_activation), 1)

        # in_shape = tuple(activation.size())
        # print(f"Frame forward: Shape = {in_shape}\r", end='')

        mean    = self.mean_out(activation)
        log_std = self.log_std_out(activation)

        # print(f"Frame forward: Shape = {in_shape} ==>")
        # print(f"       mean: Shape = {tuple(mean.size())}")
        # print(f"    log_std: Shape = {tuple(mean.size())}")
        
        return mean, log_std


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.

        Inputs:
            num_input_channels (int): Number of channels of the mask to predict. Default 1
            num_filters (int): Number of filters we use in the last convolutional
                                layers. Early layers might use a duplicate of it.
            z_dim (int): Dimensionality of latent representation z
        """
        super().__init__()

        self.residuals = []

        self.conv1a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.conv2a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(num_filters, num_input_channels, kernel_size=3, stride=1, padding=1)

        self.activation = nn.LeakyReLU()

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z_dim, 112*num_filters)
        )


    def forward(self, z):
        """
        Inputs:
            z (torch.Tensor[B,z_dim]): Latent vector
        Outputs:
            x (torch.Tensor[B, 1, 480, 854]) Predicted mask for the frame
                This should be a logit output *without* a sigmoid applied on it.
        """

        # in_shape = tuple(z.size())
        # print(f"Decode forward: Shape = {in_shape}\r", end='')

        # latent space to 2d tensor
        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 8, 14)

        # Inverse Average Pool
        x = F.interpolate(x, scale_factor=4)

        # Conv layers
        x = self.activation(self.conv1a(x) + x)
        x = self.activation(self.conv1b(x) + x)
        x = self.activation(self.conv1c(x) + x)

        # Unpad
        x = x[:,:, 1:-1, 1:-1]

        # Inverser Average Pool
        x = F.interpolate(x, scale_factor=4)

        # First residual connection
        x = x + self.residuals.pop()

        # Conv layers
        x = self.activation(self.conv2a(x) + x)
        x = self.activation(self.conv2b(x) + x)
        x = self.activation(self.conv2c(x) + x) 

        # Unpad
        x = x[:, :, :, 1:-1]

        # Inverser Average Pool
        x = F.interpolate(x, scale_factor=4)

        # second residual connection
        x = x + self.residuals.pop()

        # Conv layers
        x = self.activation(self.conv3a(x) + x)
        x = self.activation(self.conv3b(x) + x)
        x = self.conv3c(x) + x

        # print(f"Decode forward: Shape = {in_shape} ==> {tuple(x.size())}")
        
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device

    
    @torch.no_grad()
    def visualize_manifold(self, grid_size=20):
        """
        Visualize a manifold over a 2 dimensional latent space. The images in the manifold
        should represent the decoder's output means (not binarized samples of those).
        Inputs:
            decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
            grid_size - Number of steps/images to have per axis in the manifold.
                        Overall you need to generate grid_size**2 images, and the distance
                        between different latents in percentiles is 1/(grid_size+1)
        Outputs:
            img_grid - Grid of images representing the manifold.
        """
        percentiles = [norm.ppf((i + 0.5) / (grid_size+1), 0, 1) for i in range(grid_size)]
        z1, z2 = torch.meshgrid(torch.Tensor(percentiles), torch.Tensor(percentiles))

        z = torch.cat((z1.reshape(-1, 1), z2.reshape(-1, 1)), 1)
        x_mean = torch.sigmoid(self.forward(z))

        img_grid = make_grid(x_mean, nrow=grid_size)

        return img_grid



class VAE(nn.Module):

    def __init__(self, num_filters, z_dim):
        """
        PyTorch module that summarizes all components to train a VAE.
        Inputs:
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
        """
        super().__init__()
        self.z_dim = z_dim

        self.encoder = Encoder(num_filters=num_filters, z_dim=z_dim)
        self.decoder = Decoder(num_filters=num_filters, z_dim=z_dim)

    def forward(self, input_mask_batch, flow_batch, frame_batch, next_mask_batch):
        """
        The forward function calculates the VAE loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W]
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """

        # reshape the input to fit the input layer
        # input = imgs.reshape(imgs.shape[0], -1)

        # obtain mean and log std from encoder network
        mean_z, log_std_z = self.encoder.forward(input_mask_batch, flow_batch, frame_batch)
        
        # print(torch.sum(mean_z), torch.sum(log_std_z))

        # use reparameterization trick to sample a latent vector
        z = self.sample_reparameterize(mean_z, torch.exp(log_std_z))

        # use decoder to reconstruct image from latent vector
        self.decoder.residuals = self.encoder.residuals
        next_mask_predictions = self.decoder.forward(z)
        
        # compute the losses
        L_rec = F.binary_cross_entropy_with_logits(next_mask_predictions, input_mask_batch, reduction='sum')
        L_reg = torch.sum(self.KLD(mean_z, log_std_z))
        elbo = L_rec + L_reg
        bpd = self.elbo_to_bpd(elbo, input_mask_batch.shape)
        
        
        # print(f"reconstruction loss: {L_rec.item()}")
        # print(f"regularization loss: {L_reg.item()}")
        # print(f"               elbo: {elbo.item()}")
        # print(f" bits per dimension: {bpd.item()}")

        return L_rec, L_reg, bpd

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled, binarized images with 0s and 1s
            x_mean - The sigmoid output of the decoder with continuous values
                     between 0 and 1 from which we obtain "x_samples"
        """

        # sample a z vector from a unit gaussian
        z = torch.randn(size=(batch_size, self.z_dim)).to(self.device)

        # get the means of a Bernoulli distribution by passing z through the decoder
        x_mean = F.sigmoid(self.decoder.forward(z))

        # sample from a Bernoulli parameterized by the sigmoid of the decoder
        randoms = torch.rand(x_mean.size())
        x_samples = torch.where(randoms < x_mean, torch.ones(x_mean.size()), torch.zeros(x_mean.size()))
        
        return x_samples, x_mean

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.decoder.device

    def sample_reparameterize(self, mean, std):
        """
        Perform the reparameterization trick to sample from a distribution with the given mean and std
        Inputs:
            mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
            std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
                of the distribution
        Outputs:
            z - A sample of the distributions, with gradient support for both mean and std. 
                The tensor should have the same shape as the mean and std input tensors.
        """
        assert mean.size() == std.size(), f"mean and std need to have the same size. Got mean: {mean.size()}, std: {std.size()}"

        epsilon = torch.randn(mean.size()).to(self.device)
        z = mean + epsilon * std

        return z


    def KLD(self, mean, log_std):
        """
        Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
        Inputs:
            mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
            log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
        Outputs:
            KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
                The values represent the Kullback-Leibler divergence to unit Gaussians.
        """
        assert mean.size() == log_std.size(), f"mean and std need to have the same size. Got mean: {mean.size()}, log_std: {log_std.size()}"

        KLD = 0.5 * torch.sum(torch.add(torch.square(mean) + torch.square(torch.exp(log_std)) - 2 * log_std, -1), dim=-1)
        
        # for older version of torch
        # KLD = 0.5 * torch.sum(torch.add(mean * mean + torch.exp(log_std) * torch.exp(log_std)  - 2 * log_std, -1), dim=-1)
        return KLD


    def elbo_to_bpd(self, elbo, img_shape):
        """
        Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
        Inputs:
            elbo - Tensor of shape [batch_size]
            img_shape - Shape of the input images, representing [batch, channels, height, width]
        Outputs:
            bpd - The negative log likelihood in bits per dimension for the given image.
        """
        # normalize with product of image dimensions for bpd and batch dimension for mean of loss
        normalizer = prod(img_shape[1:])
        
        # Calculate bits per dimension loss
        bpd = elbo * log2(e) / normalizer
        
        return bpd