import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import cv2
import numpy as np
import random
import os

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DummieVideo(object):
    def __init__(self, frame_size, length, grid_size, noise_resolution) -> None:
        super().__init__()

        self.frame_size = frame_size
        self.length = length
        self.grid_size = grid_size
        self.noise_resolution = noise_resolution
        self.roll_speed = self.frame_size[0] // (length * 5) 

        # define different layers of the input
        self.circle = self.define_circle()
        self.grid   = self.define_grid()
        # self.noise  = self.define_noise()

        # normalize the total input to be between -1 and 1
        self.circle *= 0.33
        self.grid   *= 0.33
        # self.noise  *= 0.33

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        # select from a sliding window over the grid
        current_grid = np.roll(self.grid, idx * self.roll_speed, axis=1)[:self.frame_size[0], :self.frame_size[1]]

        self.noise = self.define_noise() * 0.33
        
        total = self.circle + current_grid + self.noise

        return total.unsqueeze(0)


    def define_circle(self, radius=None, center=None):

        w, h = self.frame_size

        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = np.where((dist_from_center <= radius), 1, -1)
        return torch.from_numpy(mask).float()

    def define_grid(self):
        w, h = self.frame_size
        padding = [w % self.grid_size, h % self.grid_size]

        X = np.arange(w + padding[0])
        Y = np.arange(h + padding[1])

        grid = torch.zeros((len(X), len(Y)))
        for i in X:
            for j in Y:
                if ((X[i] // self.grid_size) % 2 == 1) ^ ((Y[j] // self.grid_size) % 2 == 1):
                    grid[i, j] = 1
                else:
                    grid[i, j] = -1

        return grid.float()

    def define_noise(self):
        noise_root = torch.randn((self.noise_resolution, self.noise_resolution)).unsqueeze(0).unsqueeze(0)
        noise = F.interpolate(noise_root, self.frame_size, mode='bilinear', align_corners=True)
        return noise.squeeze().float()


class InputGenerator(object):
    def __init__(self, frame_size, length, noise_channels) -> None:
        super().__init__()

        scale_factor = 5
        self.frame_size = frame_size

        self.length = length
        noise_root = torch.randn((noise_channels, int(frame_size[0] / scale_factor), int(frame_size[1] / scale_factor))).unsqueeze(0)

        self.noise = F.interpolate(noise_root, size=frame_size, mode='bilinear', align_corners=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        out = torch.cat([idx*torch.ones(self.frame_size).unsqueeze(0).unsqueeze(0), self.noise], dim=1)
        return out.squeeze().float()


class DataSet(object):
    def __init__(self, frame_size, length, grid_size, noise_resolution, noise_channels) -> None:
        super().__init__()

        self.input_generator = InputGenerator(frame_size, length, noise_channels)
        self.target_generator = DummieVideo(frame_size, length, grid_size, noise_resolution)

    def __len__(self):
        return len(self.input_generator)

    def __getitem__(self, idx):
        input  = self.input_generator[idx]
        target = self.target_generator[idx] 

        return input, target, idx


class ConvBlock(nn.Module):
    """Helper module consisting of a convolution, optional normalization and activation, with padding='same'.
    
    Adapted from Layered Neural Rendering (https://github.com/google/retiming)
    """

    def __init__(self, conv, in_channels, out_channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Create a conv block.

        Parameters:
            conv (convolutional layer) - - the type of conv layer, e.g. Conv2d, ConvTranspose2d
            in_channels (int) - - the number of input channels
            in_channels (int) - - the number of output channels
            ksize (int) - - the kernel size
            stride (int) - - stride
            dil (int) - - dilation
            norm (norm layer) - - the type of normalization layer, e.g. BatchNorm2d, InstanceNorm2d
            activation (str)  -- the type of activation: relu | leaky | tanh | none
        """
        super(ConvBlock, self).__init__()
        self.k = ksize
        self.s = stride
        self.d = dil
        self.conv = conv(in_channels, out_channels, ksize, stride=stride, dilation=dil)

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        height, width = x.shape[-2:]
        if isinstance(self.conv, nn.modules.ConvTranspose2d):
            desired_height = height * self.s
            desired_width = width * self.s
            pady = 0
            padx = 0
        else:
            desired_height = height // self.s
            desired_width = width // self.s
            pady = .5 * (self.s * (desired_height - 1) + (self.k - 1) * (self.d - 1) + self.k - height)
            padx = .5 * (self.s * (desired_width - 1) + (self.k - 1) * (self.d - 1) + self.k - width)
        x = F.pad(x, [int(np.floor(padx)), int(np.ceil(padx)), int(np.floor(pady)), int(np.ceil(pady))])
        x = self.conv(x)
        if x.shape[-2] != desired_height or x.shape[-1] != desired_width:
            cropy = x.shape[-2] - desired_height
            cropx = x.shape[-1] - desired_width
            x = x[:, :, int(np.floor(cropy / 2.)):-int(np.ceil(cropy / 2.)),
                int(np.floor(cropx / 2.)):-int(np.ceil(cropx / 2.))]
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Model(nn.Module):

    def __init__(self, in_channels, conv_channels=8):
        super().__init__()

        self.layers = nn.Sequential(
            # encoder
            ConvBlock(nn.Conv2d, in_channels, conv_channels, ksize=4, stride=2, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels, conv_channels*2, ksize=4, stride=2, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels*2, conv_channels*4, ksize=4, stride=2, activation='leaky'),
            ConvBlock(nn.Conv2d, conv_channels*4, conv_channels*4, ksize=4, stride=2, activation='leaky'),
            
            # decoder
            ConvBlock(nn.ConvTranspose2d, conv_channels*4, conv_channels*4, ksize=4, stride=2),
            ConvBlock(nn.ConvTranspose2d, conv_channels*4, conv_channels*2, ksize=4, stride=2),
            ConvBlock(nn.ConvTranspose2d, conv_channels*2, conv_channels*2, ksize=4, stride=2),
            ConvBlock(nn.ConvTranspose2d, conv_channels*2, conv_channels, ksize=4, stride=2),
            
            # features -> image
            ConvBlock(nn.Conv2d, conv_channels, 1, ksize=4, stride=1, activation='tanh')
        )

        self.loss_module = nn.L1Loss()

    def forward(self, x, target):
        out = self.layers(x)

        loss = self.loss_module(out, target)

        return loss, out

#############################

def train(num_epochs, length_vid=20, frame_size=480, grid_size=48, noise_resolution=120, noise_channels=3, seed=42, lr = 1e-4, batch_size=4, vis_freq=100):

    seed_all(seed)

    # frame size is a square
    frame_size = [frame_size]*2 

    dataset = DataSet(frame_size, length_vid, grid_size, noise_resolution, noise_channels)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    model = Model(noise_channels+1)

    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        print(f"{epoch} / {num_epochs}", end="\r")
        for (input, target, frame_idx) in dataloader:

            optimizer.zero_grad()
            loss, output = model.forward(input, target)
            loss.backward()
            optimizer.step()
        
            if epoch % vis_freq == 0:
                visualize_output(output, target, frame_idx, epoch)
        print(f"{epoch} / {num_epochs} | done")

def visualize_output(output, target, frame_indices, epoch):

    for b in range(output.size(0)):
        frame_idx = frame_indices[b]
        output_img = (cv2.resize(output[b].clone().detach().permute(1, 2, 0).numpy(), (480, 480)) * .5 + .5) * 255
        target_img = (cv2.resize(target[b].clone().detach().permute(1, 2, 0).numpy(), (480, 480)) * .5 + .5) * 255

        if epoch == 0:
            outdir = f"experiments/results/spatiotemporal_noise/targets"
            os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(f"{outdir}/{frame_idx:05}.png", target_img)

        outdir = f"experiments/results/spatiotemporal_noise/epoch_{epoch:05}"
        os.makedirs(outdir, exist_ok=True)
        cv2.imwrite(f"{outdir}/{frame_idx:05}.png", output_img)

if __name__ == "__main__":
    train(10001)