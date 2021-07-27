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
    def __init__(self, num_filters: int = 32, z_dim: int = 20, batch_size: int = 1):
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
        self.flow_encoder  =  FlowEncoder(num_filters=num_filters, z_dim=z_dim)
        self.frame_encoder = FrameEncoder(num_filters=num_filters, z_dim=z_dim)

        self.lstm = nn.LSTM(3*z_dim, z_dim)
        self.lstm_hidden_state = torch.zeros(1, batch_size, z_dim).to(self.device)
        self.lstm_cell_state   = torch.zeros(1, batch_size, z_dim).to(self.device)

        self.mean_out    = nn.Linear(z_dim, z_dim)
        self.log_std_out = nn.Linear(z_dim, z_dim)
        
        self.z_dim = z_dim
        self.batch_size = batch_size

    def forward(self, mask, flow, frame):
        """
        Forward pass of the encoder model. Runs each input through their respective encoder model 
        and combines the results in one latent vector

        Inputs:
            mask_seq  (torch.Tensor[B, 1, 480, 856]): The binary mask defined on the current frame
            flow_seq  (torch.Tensor[B, 2, 480, 856]): The optical flow between the current frame and the next
            frame_seq (torch.Tensor[B, 3, 480, 856]): The image data of the current frame

        Outputs:
            mean    (torch.Tensor[B, z_dim]): A row vector of diminsions {z_dim} for the mean of the latent 
                                                distribution
            log_std (torch.Tensor[B, z_dim]): A row vector of diminsions {z_dim} for the log standard deviation
                                                of the latent distribution
        """

        mask_activation  = self.mask_encoder.forward(mask)
        flow_activation  = self.flow_encoder.forward(flow)
        frame_activation = self.frame_encoder.forward(frame)

        activation = torch.cat((mask_activation, flow_activation, frame_activation), 1).unsqueeze(0)

        activation, (self.lstm_hidden_state, self.lstm_cell_state) = self.lstm(activation, \
                                                                    (self.lstm_hidden_state, self.lstm_cell_state))

        activation = activation.squeeze(0)

        mean = self.mean_out(activation)
        log_std = self.log_std_out(activation)
        
        return mean, log_std

    def reset_lstm(self):
        """
        Reset the hidden state and cell state of the LSTM module in the encoder
        """
        self.lstm_hidden_state = torch.zeros(1, self.batch_size, self.z_dim).to(self.device)
        self.lstm_cell_state   = torch.zeros(1, self.batch_size, self.z_dim).to(self.device)
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.lstm_hidden_state = self.lstm_hidden_state.to(*args, **kwargs)
        self.lstm_cell_state = self.lstm_cell_state.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return next(self.parameters()).device

class Decoder(nn.Module):
    def __init__(self, num_filters: int = 32, z_dim: int = 20):
        """
        Decoder with a CNN network.

        Inputs:
            num_input_channels (int): Number of channels of the mask to predict. Default 1
            num_filters (int): Number of filters we use in the last convolutional
                                layers. Early layers might use a duplicate of it.
            z_dim (int): Dimensionality of latent representation z
        """
        super().__init__()

        self.conv1a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv1c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.conv2a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv2c = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.conv3c = nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1)

        self.activation = nn.LeakyReLU()

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(z_dim, 112*num_filters)
        )


    def forward(self, z):
        """
        Inputs:
            z (torch.Tensor[T, B, z_dim]): Latent vector
        Outputs:
            x (torch.Tensor[T, B, 1, 480, 854]) Predicted mask for the frame
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

        # Conv layers
        x = self.activation(self.conv2a(x) + x)
        x = self.activation(self.conv2b(x) + x)
        x = self.activation(self.conv2c(x) + x) 

        # Unpad
        x = x[:, :, :, 1:-1]

        # Inverser Average Pool
        x = F.interpolate(x, scale_factor=4)

        # Conv layers
        x = self.activation(self.conv3a(x) + x)
        x = self.activation(self.conv3b(x) + x)
        x = self.conv3c(x)

        # print(f"Decode forward: Shape = {in_shape} ==> {tuple(x.size())}")
        
        return x

    @property
    def device(self):
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



class RecurrentMaskPropVAE(nn.Module):

    def __init__(self, num_filters, z_dim, batch_size):
        """
        PyTorch module that summarizes all components to train a VAE.
        Inputs:
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
        """
        super().__init__()
        self.z_dim = z_dim

        self.encoder = Encoder(num_filters=num_filters, z_dim=z_dim, batch_size=batch_size)
        self.decoder = Decoder(num_filters=num_filters, z_dim=z_dim)

    def forward(self, mask_seq, flow_seq, frame_seq):
        """
        forward training function that calculates the loss of the model by sequentially predicting
        a new mask based on the previous predicted mask and the rgb values and optical flow from the images

        Args:
            mask_batch  (torch.Tensor[T, B, 1, W, H]): The ground truth masks of the sequence of images
            flow_batch  (torch.Tensor[T, B, 2, W, H]): A batch of the optical flow estimations between the frames in the sequence
            frame_batch (torch.Tensor[T, B, 2, W, H]): The rgb values of the image sequence.=

        Returns:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """

        # get the sequence length from the input
        seq_length = mask_seq.size()[1]
        
        mask = mask_seq[:, 0, :, :, :]       

        L_rec = []
        L_reg = []

        # reset internal state of recurrent network
        self.encoder.reset_lstm()

        for ts in range(seq_length - 1):

            # prepare input for current time step
            flow  = flow_seq[:, ts, :, :, :]
            frame = frame_seq[:, ts + 1, :, :, :]

            # obtain mean and log std from encoder network
            mean_z, log_std_z = self.encoder.forward(mask, flow, frame)

            # use reparameterization trick to sample a latent vector
            z = self.sample_reparameterize(mean_z, torch.exp(log_std_z))
        
            # use decoder to reconstruct image from latent vector
            next_mask_predictions = self.decoder.forward(z)
            mask = torch.clone(nn.Sigmoid()(next_mask_predictions))
        
            gt_mask = mask_seq[:, ts + 1, :, :, :]
            # compute the losses
            if ts == 0:            
                L_rec = F.binary_cross_entropy_with_logits(next_mask_predictions, gt_mask, reduction='sum')
                L_reg = torch.sum(self.KLD(mean_z, log_std_z))
                neg_elbo = L_rec + L_reg
                bpd = self.neg_elbo_to_bpd(neg_elbo, gt_mask.shape)
            else:
                L_rec += F.binary_cross_entropy_with_logits(next_mask_predictions, gt_mask, reduction='sum')
                L_reg += torch.sum(self.KLD(mean_z, log_std_z))
                neg_elbo += L_rec + L_reg
                bpd += self.neg_elbo_to_bpd(neg_elbo, gt_mask.shape)                

        return L_rec, L_reg, bpd


    def predict_next_mask(self, current_mask, flow, frame):

        # obtain mean and log std from encoder network
        mean_z, log_std_z = self.encoder.forward(current_mask, flow, frame)
        
        # use reparameterization trick to sample a latent vector
        z = self.sample_reparameterize(mean_z, torch.exp(log_std_z))

        # use decoder to reconstruct image from latent vector
        next_mask_prediction = self.decoder.forward(z)
        next_mask_prediction = nn.Sigmoid()(next_mask_prediction)

        return next_mask_prediction


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

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        return self

    @property
    def device(self):
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


    def neg_elbo_to_bpd(self, neg_elbo, img_shape):
        """
        Converts the summed negative log likelihood given by the negative ELBO into the bits per dimension score.
        Inputs:
            neg_elbo - Tensor of shape [batch_size]
            img_shape - Shape of the input images, representing [batch, channels, height, width]
        Outputs:
            bpd - The negative log likelihood in bits per dimension for the given image.
        """
        # normalize with product of image dimensions for bpd and batch dimension for mean of loss
        normalizer = prod(img_shape[1:])
        
        # Calculate bits per dimension loss
        bpd = neg_elbo * log2(e) / normalizer
        
        return bpd