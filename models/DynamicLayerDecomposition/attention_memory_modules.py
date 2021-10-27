import torch
import torch.nn as nn
from InputProcessing.inputProcessor import InputProcessor
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

    def __init__(self, keydim: int, valdim: int, backbone) -> None:
        super().__init__()

        self.backbone = backbone

        indim = 1024 # defined by ResNet50

        self.key_layer = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)
        self.val_layer = nn.Conv2d(indim, valdim, kernel_size=3, padding=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # the backbone will not be trained
        with torch.no_grad():
            x = self.backbone(x)

        if isinstance(x, tuple):
            x = x[0]

        key = self.key_layer(x)
        val = self.val_layer(x)
        
        return key, val


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

    def __init__(self, keydim: int, valdim: int, num_layers: int, mem_freq: int, input_processor: InputProcessor, backbone_weights: str) -> None:
        super().__init__()

        # Create a memory for every object layer plus one for the background layer
        self.memory_encoders = nn.ModuleList([KeyValueEncoder(keydim, valdim, create_backbone(backbone_weights, with_masks=True))] * num_layers)
        self.global_contexts = nn.ModuleList([GlobalContextVolume(keydim, valdim)] * num_layers)

        self.input_processor = input_processor
        self.frame_iterator  = input_processor.frame_iterator
        self.mask_iterator   = input_processor.mask_handler
        self.num_layers      = num_layers
        self.mem_freq        = mem_freq
        self.valdim          = valdim
        self.keydim          = keydim
        self.device          = self.frame_iterator.device

    def set_global_contexts(self, jitter_params) -> None:
        """
        Set the global context volumes from a set. This is equivalent to memorizing frames in a memory network
        but because we are overfitting on a single video we do this with a number of frames at the start of each epoch
        """
    
        # update the global context for each object layer separately
        for l in range(self.num_layers):
            local_contexts = []

            # create iterator for containing all frames that need to be considered
            iterator = list(range(0, len(self.frame_iterator), self.mem_freq))
            # if iterator[-1] < len(self.frame_iterator):
            #     iterator.append(len(self.frame_iterator) - 1)

            # add frames to memory
            for frame_idx in iterator:

                # get frame and add batch dimension
                frame = self.frame_iterator[frame_idx].unsqueeze(0).to(self.device)
                
                # get mask and add batch dimension
                _, object_mask = self.mask_iterator[frame_idx]
                object_mask = object_mask.to(self.device)

                # apply jitter
                frame       = self.input_processor.apply_jitter_transform(frame, jitter_params[frame_idx])
                object_mask = self.input_processor.apply_jitter_transform(object_mask, jitter_params[frame_idx])

                # create mask layer for other object layers
                background_mask = torch.zeros_like(object_mask)

                if l == 0:
                    mask  = background_mask
                    other = object_mask
                else:
                    mask  = object_mask
                    other = background_mask

                frame = torch.cat((frame, mask, other), dim=1)

                # encode frame and get context matrix
                encoded_frame = self.memory_encoders[l](frame)
                context = self.get_context_from_key_value_pair(encoded_frame)

                # remove batch dimension
                context = context.squeeze(0)

                # append to collection of local contexts of current layer
                local_contexts.append(context)

            # update the memory of the current layer
            self.global_contexts[l].update(local_contexts)

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
    def __init__(self, keydim: int, valdim: int, num_layers: int, backbone_weights: str) -> None:
        super().__init__()

        self.query_encoders = nn.ModuleList([KeyValueEncoder(keydim, valdim, create_backbone(backbone_weights, with_masks=False))]*num_layers)

        self.num_layers = num_layers
        self.keydim = keydim
        self.valdim = valdim

    def forward(self, query_imgs: torch.Tensor, object_idx: int, global_context: GlobalContextVolume) -> torch.Tensor:
        """
        Get the batch of ground truth frames and read from the current object memory using the frames as query

        Args:
            query_imgs (torch.Tensor[B x 3 x H x W])
            object_idx (int)

        Returns:
            feature_map (torch.Tensor[B x 2 * C_v x H x W])
        """

        query, value = self.query_encoders[object_idx](query_imgs)

        global_features = global_context(query)

        feature_map = torch.cat((global_features, value), dim=1)
        return feature_map