import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Helper module consisting of a convolution, optional normalization and activation, with padding='same'.
    
    Adapted from Layered Neural Rendering (https://github.com/google/retiming)
    """

    def __init__(self, out_channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Create a conv block.

        Parameters:
            conv (convolutional layer) - - the type of conv layer, e.g. Conv2d, ConvTranspose2d
            in_channels (int) - - the number of input channels
            out_channels (int) - - the number of output channels
            ksize (int) - - the kernel size
            stride (int) - - stride
            dil (int) - - dilation
            norm (norm layer) - - the type of normalization layer, e.g. BatchNorm2d, InstanceNorm2d
            activation (str)  -- the type of activation: relu | leaky | tanh | none
        """
        super().__init__()
        self.k = ksize
        self.s = stride
        self.d = dil

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
        elif activation == 'channel_softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        return NotImplemented


class GlobalContextVolume(nn.Module):
    """
    Global Context Volume base class

    Takes care of all the computations on the context distributions and keeps a copy of the current context distribution
    in memory
    """

    def __init__(self, keydim: int, valdim: int, topk: int = 0) -> None:
        super().__init__()

        # self.register_buffer(f"context_volume", torch.zeros(valdim, keydim))
        self.topk = topk if topk > 0 else None

        # for running average
        self.step = 1

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Returns a context distribution defined by the global context and the local query

        D_t = q(x_t) * G

        Args:
            query (torch.Tensor[B, C_k, T, H, W])

        Returns:
            context_dist (torch.Tensor[B, C_v, T, H, W])

        """
        return NotImplemented

    def update(self, context: torch.Tensor) -> None:
        """
        Update the global context volume using the local context matrices at different timesteps
        v1:
            the average of the local context matrices is taken
        
        Args:
            local_contexts (list[ torch.Tensor[C_v x C_k] ] -- length=T)
        """

        if self.step == 1:
            self.context_volume = context
        else:
            step = self.step
            self.context_volume = (step - 1) / step * self.context_volume + 1 / step * context
        self.step += 1

    def reset_steps(self):
        """
        Reset the self.step variable to start a new running average
        """
        self.step = 1

class MemoryEncoder(nn.Module):
    """
    Memory Encoder base class

    Encodes the input into a global context
    """

    def __init__(self, keydim: int) -> None:
        super().__init__()

        self.key_layer      = NotImplemented
        self.value_layer    = NotImplemented
        self.global_context = NotImplemented

        self.keydim    = keydim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Update the global context distribution
        """

        key = self.key_layer(input)
        val = self.value_layer(input)

        context = self._get_context_from_key_value_pair(key, val)

        # update the memory of the current layer 
        self.global_context.update(context[0])

        return key

    def _get_context_from_key_value_pair(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Takes as argument a key value pair and returns the corresponding context matrix

        The context is calculated according to:
            C [B x C_v x C_k] = v [B x C_v x (T)HW] @ k [B x (T)HW x C_k] 

        Args:
            key   (torch.Tensor)
            value (torch.Tensor)

        Returns:
            context (torch.Tensor)
        """
        return NotImplemented
