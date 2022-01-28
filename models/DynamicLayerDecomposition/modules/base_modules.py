import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat


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
        else:
            self.activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        return NotImplemented


class KeyValueEncoder(nn.Module):
    """
    Key-Value encoder network base class
    
    Usable with both 2D and 3D convolutions
    """
    def __init__(self, conv_module: nn.Module, conv_channels: int, keydim: int, valdim: int) -> None:
        super().__init__()

        self.key_layer = conv_module(conv_channels, keydim, kernel_size=4, padding='same')
        self.val_layer = conv_module(conv_channels, valdim, kernel_size=4, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through the network

        Args:
            x (torch.Tensor)

        Returns:
            key (torch.Tensor): key tensor
            val (torch.Tensor): value tensor
            skips (list[torch.Tensor]): list of skip connections
        """
        
        key = F.softmax(self.key_layer(x), dim=1)
        val = F.leaky_relu(self.val_layer(x), negative_slope=0.2)
        
        return key, val


class GlobalContextVolume(nn.Module):
    """
    Global Context Volume base class

    Takes care of all the computations on the context distributions and keeps a copy of the current context distribution
    in memory
    """

    def __init__(self, keydim: int, valdim: int, n_layers: int, topk: int = 0) -> None:
        super().__init__()

        self.context_volume = [torch.zeros(valdim, keydim)]*n_layers
        self.topk = topk if topk > 0 else None

        # for running average
        self.n_layers = n_layers
        self.step = list(repeat(1, n_layers))

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

    def update(self, context: torch.Tensor, layer_idx) -> None:
        """
        Update the global context volume using the local context matrices at different timesteps
        v1:
            the average of the local context matrices is taken
        
        Args:
            local_contexts (list[ torch.Tensor[C_v x C_k] ] -- length=T)
        """
        if context.device != self.context_volume[layer_idx].device:
            context.to(self.context_volume[layer_idx].device)

        if self.step[layer_idx] == 1:
            self.context_volume[layer_idx] = context
        else:
            step = self.step[layer_idx]
            self.context_volume[layer_idx] = (step - 1) / step * self.context_volume[layer_idx] + 1 / step * context
        self.step[layer_idx] += 1

    def reset_steps(self):
        """
        Reset the self.step variable to start a new running average
        """
        self.step = list(repeat(1, self.n_layers))

class MemoryEncoder(nn.Module):
    """
    Memory Encoder base class

    Encodes the input into a global context
    """

    def __init__(self, keydim: int, reconstruction_encoder: nn.ModuleList) -> None:
        super().__init__()

        self.backbone       = reconstruction_encoder
        self.key_layer      = NotImplemented
        self.value_layer    = NotImplemented
        self.global_context = NotImplemented

        # self.valdim    = valdim
        self.keydim    = keydim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Update the global context distribution
        """

        feature_maps = []
        with torch.no_grad():
            for l in range(input.shape[1]):
                x = input[:, l]
                for layer in self.backbone:
                    x = layer(x)

                feature_maps.append(x)

        for l in range(len(feature_maps)):

            key = F.softmax(self.key_layer(feature_maps[l]), dim=1)
            val = F.leaky_relu(self.value_layer(x))

            context = self._get_context_from_key_value_pair(key, val)

            # update the memory of the current layer
            for b in range(context.shape[0]):
                self.global_context.update(context[b], l)

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


# class MemoryReader(nn.Module):
#     """
#     Memory Reader base class

#     Encodes the input into a key value pair. the key is used as query to retrieve a feature map from the 
#     global context distribution. The value tensor, feature map and skip connections are passed on.

#     Usable with both 2D and 3D convolutions
#     """
#     def __init__(self, conv_module: nn.Module, conv_channels: int, keydim: int, valdim: int, backbone: nn.Module) -> None:
#         super().__init__()

#         self.query_encoder = KeyValueEncoder(conv_module, conv_channels, keydim, valdim, backbone)

#         self.keydim = keydim
#         self.valdim = valdim

#     def forward(self, input: torch.Tensor, global_context: GlobalContextVolume) -> torch.Tensor:
#         """
#         Args:
#             input (torch.Tensor)

#         Returns:
#             context_distribution (torch.Tensor)
#             value (torch.Tensor)
#             skips (list[torch.Tensor])
#         """
#         query, value, skips = self.query_encoder(input)

#         context_distribution = global_context(query)

#         return context_distribution, value, skips
