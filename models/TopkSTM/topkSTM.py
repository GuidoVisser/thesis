from typing import Union
import torch
from torch import nn

from .modules.prop_net import PropagationNetwork
from .utils.utils import pad_divide_by, aggregate_wbg


# TODO make the class compatible with DataParallel
class TopKSTM(nn.Module):
    def __init__(self, memory_size:int, model_device, memory_device: str="cpu", top_k: int=50, mem_freq: int=5):
        super().__init__()

        self.device = model_device

        self.prop_net = PropagationNetwork(top_k).to(model_device)
        self.mem_freq = mem_freq
        
        self.key_memory = Memory(memory_size, memory_device, model_device)
        self.val_memory = Memory(memory_size, memory_device, model_device)

    def predict_mask_and_memorize(self, frame_idx:int, frame: torch.Tensor, frame_divide_by: int=16) -> torch.Tensor:
        """
        Predict the mask of a frame given the memory that has already been acccumulated and add this mask to the memory

        Args:
            frame_idx (int): denotes how far into a sequency we are
            frame (torch.Tensor[K, B, CF, H, W])

        Returns:
            mask (torch.Tensor[K, B, CM, H, W])
        """
        # prepare frame
        frame = pad_divide_by(frame, frame_divide_by).to(self.device)

        # get keys, values and query
        keys = self.key_memory.get()
        vals = self.val_memory.get()
        query = self.prop_net.get_query_values(frame)

        # generate segmentation
        mask = self.prop_net.segment_with_query(keys, vals, *query)
        mask = aggregate_wbg(mask)

        # free up memory on device
        del keys, vals, query

        # memorize results
        extend_memory = True if frame_idx % self.mem_freq == 0 else False
        self.add_to_memory(frame, mask, extend_memory)

        return mask

    def add_to_memory(self, frame: torch.Tensor, mask: torch.Tensor, extend_memory: bool=False) -> None:
        """
        generate a key and value pair based on a frame and mask and update the memory

        << Updating memory >>
        either:
            expand the memory every if extend_memory is True
        or:
            replace the last entry of the memory otherwise

        Args:
            TODO check frame and mask dimensions
            frame (torch.Tensor[B, K, CF, H, W])
            mask (torch.Tensor[B, K, CM, H, W])
            extend_memory (bool): denotes whether to replace the last entry in memory or extend the memory
        """
       
        # get new key and val from frame and mask
        new_key, new_val = self.prop_net.memorize(frame, mask)

        # add new key and val to memory
        self.key_memory.add_entry(new_key, extend_memory=extend_memory)
        self.val_memory.add_entry(new_val, extend_memory=extend_memory)

    def reset_memory(self) -> None:
        self.key_memory.reset()
        self.val_memory.reset()


class Memory(object):

    def __init__(self, size: int, memory_device: str, model_device: str) -> None:
        super().__init__()
        self.front_pointer = 0
        self.data = None
        self.size = size
        self.memory_device = memory_device
        self.model_device = model_device

    def get(self):
        return self.data[:, :, :self.front_pointer+1].to(self.model_device)

    def add_entry(self, entry: torch.Tensor, extend_memory: bool=False) -> None:
        if len(entry.size()) != 5:
            raise ValueError("Shape of entry is incorrect, it has to be [B, C, T, H, W]")

        if self.data == None:
            B, C, _, H, W = entry.size()
            self.data = torch.empty((B, C, self.size, H, W), dtype=torch.float32, device=self.memory_device)
            
        # self[self.front_pointer] = entry.to(self.memory_device)
        self._data[:, :, self.front_pointer] = entry.to(self.memory_device)

        if extend_memory:
            self.front_pointer += 1

    def reset(self) -> None:
        self.data = torch.empty(self.data.size()).to(self.memory_device)

    def __len__(self) -> int:
        if self.data == None:
            return 0
        else:
            return self.front_pointer
    
    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        self._size = value
    
    @property
    def data(self) -> Union[torch.Tensor, None]:
        return self._data

    @data.setter
    def data(self, value: Union[torch.Tensor, None]) -> None:
        if type(value) == torch.Tensor:
            if not len(value.size()) == 5:
                raise ValueError("Shape of memory is incorrect, it has to be [B, C, T, H, W]")
        
        self._data = value

    @property
    def front_pointer(self) -> int:
        return self._front_pointer

    @front_pointer.setter
    def front_pointer(self, value: int) -> None:
        if value > self.size: 
            raise ValueError(f"Front of memory at {value} is out of bounds for memory size of {self.size}")
        self._front_pointer = value