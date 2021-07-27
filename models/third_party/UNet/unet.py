import torch.nn as nn
import torch
from .unet_blocks import EncoderBlock, DecoderBlock

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encode1 = EncoderBlock(in_channels, 32, 7, 3)
        self.encode2 = EncoderBlock(32, 64, 3, 1)
        self.encode3 = EncoderBlock(64, 128, 3, 1)

        self.decode1 = DecoderBlock(128, 64, 3, 1)
        self.decode2 = DecoderBlock(64*2, 32, 3, 1)
        self.decode3 = DecoderBlock(32*2, out_channels, 3, 1)

    def __call__(self, x):

        encoding1 = self.encode1(x)
        encoding2 = self.encode2(encoding1)
        encoding3 = self.encode3(encoding2)

        decode1 = self.decode1(encoding3)
        decode2 = self.decode2(torch.cat([decode1, encoding2], 1)) 
        decode3 = self.decode3(torch.cat([decode2, encoding1], 1))

        return decode3
