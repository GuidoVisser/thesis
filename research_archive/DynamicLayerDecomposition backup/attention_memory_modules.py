import torch
import torch.nn as nn
from InputProcessing.inputProcessor import InputProcessor
from models.DynamicLayerDecomposition.modules import ConvBlock
# from torchvision.models import resnet50
from models.TopkSTM.modules.modules import MaskRGBEncoder, RGBEncoder

def create_backbone(weights_path: str, with_masks: bool):

    model_weights = torch.load(weights_path)

    if with_masks:
        model = MaskRGBEncoder()
        name = "mask_rgb_encoder"
    else:
        model = RGBEncoder()
        name = "rgb_encoder"

    new_dict = {}
    for key, value in model_weights.items():
        if name in key and not (("mask" in key) != ("mask" in name)):
            new_key = ".".join(key.split(".")[1:])
            new_dict[new_key] = value
    model.load_state_dict(new_dict)
    model.eval()

    return model


# class ResNet50Backbone(nn.Module):

#     def __init__(self):
#         super().__init__()
        
#         resnet = resnet50(pretrained=True)
#         self.conv1   = resnet.conv1
#         self.bn1     = resnet.bn1
#         self.relu    = resnet.relu  
#         self.maxpool = resnet.maxpool

#         self.res2   = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3

#         self.out_channels = 1024

#     def forward(self, f: torch.Tensor):
#         x   = self.conv1(f) 
#         x   = self.bn1(x)
#         x   = self.relu(x)     # 1/2, 64
#         x   = self.maxpool(x)  # 1/4, 64
#         f4  = self.res2(x)     # 1/4, 256
#         f8  = self.layer2(f4)  # 1/8, 512
#         f16 = self.layer3(f8)  # 1/16, 1024

#         return f16


class KeyValueEncoder(nn.Module):

    def __init__(self, in_channels: int, conv_channels: int, keydim: int, valdim: int) -> None:
        super().__init__()

        self.encoder = nn.ModuleList([
            ConvBlock(nn.Conv2d, in_channels,       conv_channels,     ksize=4, stride=2),                                           # 1/2
            ConvBlock(nn.Conv2d, conv_channels,     conv_channels * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/4
            ConvBlock(nn.Conv2d, conv_channels * 2, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'),  # 1/8
            ConvBlock(nn.Conv2d, conv_channels * 4, conv_channels * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky')]) # 1/16

        self.key_layer = nn.Conv2d(conv_channels * 4, keydim, kernel_size=3, padding=1)
        self.val_layer = nn.Conv2d(conv_channels * 4, valdim, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)

        key = self.key_layer(x)
        val = self.val_layer(x)
        
        return key, val, skips


class GlobalContextVolume(nn.Module):
    def __init__(self, keydim: int, valdim: int) -> None:
        super().__init__()

        self.register_buffer("context_volume", torch.zeros((valdim, keydim)))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Returns a context distribution defined by the global context and the local query

        D_t = q(x_t) * G

        Args:
            query (torch.Tensor[B x C_k x H, W])

        Returns:
            context_dist (torch.Tensor[B x C_v x H x W])

        """
        B, _, H, W = query.shape

        query = query.view(B, -1, H*W)                          # -> [B x C_k x HW]
        context_dist = torch.matmul(self.context_volume, query) # -> [B x C_v x HW]
        context_dist = context_dist.view(B, -1, H, W)           # -> [B x C_v x H x W]

        return context_dist

    def update(self, local_contexts: list) -> None:
        """
        Update the global context volume using the local context matrices at different timesteps
        v1:
            the average of the local context matrices is taken
        
        Args:
            local_contexts (list[ torch.Tensor[C_v x C_k] ] -- length=T)
        """
        self.context_volume = torch.mean(torch.stack(local_contexts, dim=0), dim=0, keepdim=False)


class AttentionMemoryNetwork(nn.Module):

    def __init__(self, in_channels: int, conv_channels: int, keydim: int, valdim: int, input_processor: InputProcessor) -> None:
        super().__init__()

        self.memory_encoder = KeyValueEncoder(in_channels, conv_channels, keydim, valdim)
        self.global_context = GlobalContextVolume(keydim, valdim)

        self.input_processor = input_processor
        self.valdim          = valdim
        self.keydim          = keydim
        self.device          = self.input_processor.device

    def set_global_contexts(self, jitter_params) -> None:
        """
        Set the global context volumes from a set. This is equivalent to memorizing frames in a memory network
        but because we are overfitting on a single video we do this with a number of frames at the start of each epoch
        """
    
        local_contexts = []

        spatiotemporal_noise = self.input_processor.background_volume.spatiotemporal_noise.to(self.device)

        # add frames to memory
        for frame_idx in range(spatiotemporal_noise.shape[1]):

            input = spatiotemporal_noise[:, frame_idx].unsqueeze(0)

            # apply jitter
            input = self.input_processor.apply_jitter_transform(input, jitter_params[frame_idx])

            # encode frame and get context matrix
            key, value, _ = self.memory_encoder(input)
            context = self.get_context_from_key_value_pair((key, value))

            # remove batch dimension
            context = context.squeeze(0)

            # append to collection of local contexts of current layer
            local_contexts.append(context)

        # update the memory of the current layer
        self.global_context.update(local_contexts)

    def get_context_from_key_value_pair(self, pair: tuple) -> torch.Tensor:
        """
        Takes as argument a key value pair and returns the corresponding context matrix

        The context is calculated according to:
            C [B x C_v x C_k] = v [B x C_v x HW] @ k [B x HW x C_k] 

        Args:
            pair (tuple[Tensor[B x C_k x H x W], Tensor[B x C_v x H x W]]): key value pairs

        Returns:
            context (Tensor[B x C_v x C_k])

        """
        key, value = pair

        B, _, H, W = key.shape
        
        key   = key.view(B, -1, H*W).permute(0, 2, 1)
        value = value.view(B, -1, H*W)

        context = torch.matmul(value, key)

        return context


class MemoryReader(nn.Module):
    def __init__(self, in_channels: int, conv_channels: int, keydim: int, valdim: int) -> None:
        super().__init__()

        self.query_encoder = KeyValueEncoder(in_channels, conv_channels, keydim, valdim)

        self.keydim = keydim
        self.valdim = valdim

    def forward(self, input: torch.Tensor, global_context: GlobalContextVolume) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor[B x C x H x W])

        Returns:
            feature_map (torch.Tensor[B x 2 * C_v x H x W])
        """

        query, value, skips = self.query_encoder(input)

        global_features = global_context(query)

        feature_map = torch.cat((global_features, value), dim=1)

        return feature_map, skips


# class ContextDecoder(nn.Module):
#     def __init__(self, valdim, out_channels):
#         super().__init__()

#         self.layers = nn.ModuleList([
#             ConvBlock(nn.ConvTranspose2d, valdim * 2,        out_channels * 64, ksize=4, stride=2),
#             ConvBlock(nn.ConvTranspose2d, out_channels * 64, out_channels * 16, ksize=4, stride=2),
#             ConvBlock(nn.ConvTranspose2d, out_channels * 16, out_channels * 4,  ksize=4, stride=2),
#             ConvBlock(nn.ConvTranspose2d, out_channels * 4,  out_channels,      ksize=4, stride=2)
#         ])

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x