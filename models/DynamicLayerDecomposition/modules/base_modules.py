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
    def __init__(self, conv_module: nn.Module, conv_channels: int, keydim: int, valdim: int, encoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder

        self.key_layer = conv_module(conv_channels, keydim, kernel_size=3, padding=1)
        self.val_layer = conv_module(conv_channels, valdim, kernel_size=3, padding=1)

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
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 4:
                skips.append(x)

        key = self.key_layer(x)
        val = self.val_layer(x)
        
        return key, val, skips


class GlobalContextVolume(nn.Module):
    """
    Global Context Volume base class

    Takes care of all the computations on the context distributions and keeps a copy of the current context distribution
    in memory
    """

    def __init__(self, keydim: int, valdim: int) -> None:
        super().__init__()

        self.register_buffer("context_volume", torch.zeros((valdim, keydim)))

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

    def update(self, local_contexts: list) -> None:
        """
        Update the global context volume using the local context matrices at different timesteps
        v1:
            the average of the local context matrices is taken
        
        Args:
            local_contexts (list[ torch.Tensor[C_v x C_k] ] -- length=T)
        """
        self.context_volume = torch.mean(torch.stack(local_contexts, dim=0), dim=0, keepdim=False)


class MemoryEncoder(nn.Module):
    """
    Memory Encoder base class

    Encodes the input into a global context
    """

    def __init__(self, keydim: int, valdim: int) -> None:
        super().__init__()

        self.memory_encoder = NotImplemented
        self.global_context = NotImplemented

        self.valdim    = valdim
        self.keydim    = keydim

    def forward(self, spatiotemporal_noise: torch.Tensor) -> GlobalContextVolume:
        """
        Set the global context volume for this epoch
        """
    
        local_contexts = []

        # construct an iterator
        frame_iterator = self._get_frame_idx_iterator(spatiotemporal_noise.shape[-3])

        for frame_idx in frame_iterator:

            # get current input
            input = self._get_input(spatiotemporal_noise, frame_idx)

            # encode frame and get context matrix
            key, value, _ = self.memory_encoder(input)
            context = self._get_context_from_key_value_pair(key, value)

            # remove batch dimension
            context = context.squeeze(0)

            # append to collection of local contexts of current layer
            local_contexts.append(context)

        # update the memory of the current layer
        self.global_context.update(local_contexts)

        return self.global_context

    def _get_input(self, spatiotemporal_noise: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        Get the input for a single timestep to add to the global context

        Args:
            spatiotemoral_noise (torch.Tensor): noise tensor of the entire video
            frame_idx (int): index of first frame in the input
        """
        return NotImplemented

    def _get_frame_idx_iterator(self, length_video: int):
        """
        Get a range object that specifies which frame indices should be considered in the memory

        Args:
            length_video (int): number of frames in the video        
        """
        return NotImplemented


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


class MemoryReader(nn.Module):
    """
    Memory Reader base class

    Encodes the input into a key value pair. the key is used as query to retrieve a feature map from the 
    global context distribution. The value tensor, feature map and skip connections are passed on.

    Usable with both 2D and 3D convolutions
    """
    def __init__(self, conv_module: nn.Module, conv_channels: int, keydim: int, valdim: int, backbone: nn.Module) -> None:
        super().__init__()

        self.query_encoder = KeyValueEncoder(conv_module, conv_channels, keydim, valdim, backbone)

        self.keydim = keydim
        self.valdim = valdim

    def forward(self, input: torch.Tensor, global_context: GlobalContextVolume) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor)

        Returns:
            context_distribution (torch.Tensor)
            value (torch.Tensor)
            skips (list[torch.Tensor])
        """
        query, value, skips = self.query_encoder(input)

        context_distribution = global_context(query)

        return context_distribution, value, skips
