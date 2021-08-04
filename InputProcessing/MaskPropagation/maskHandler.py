import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from PIL import Image
from os import path, listdir
from typing import Union
import torch.nn.functional as F

from utils.utils import create_dir
from utils.video_utils import save_frame
from utils.transforms import get_transforms
from models.TopkSTM.topkSTM import TopKSTM
from models.TopkSTM.utils import pad_divide_by
from datasets import Video

class MaskHandler(object):
    def __init__(self,
                 img_dir: str,
                 mask_dir: str,
                 initial_masks: Union[str, list],
                 frame_size: list,
                 device: str = "cuda",
                 binary_threshold: float = 0.5) -> None:
        super().__init__()

        # set hyperparameters
        self.frame_size             = frame_size
        self.binary_threshold = binary_threshold

        # set up directories and correct input for masks
        if isinstance(initial_masks, str):
            initial_masks = [initial_masks]
        
        self.N_objects = len(initial_masks)
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.device    = device

        # propagate each object mask through video
        for i in range(self.N_objects):
            save_dir = path.join(self.mask_dir, f"{i:02}")
            create_dir(save_dir)
            self.propagate(initial_masks[i], 50, 10, "cuda", "cuda", "models/third_party/weights/propagation_model.pth", save_dir)
         
    @torch.no_grad()
    def propagate(self, initial_mask, top_k, mem_freq, model_device, memory_device, model_weights, save_dir):
        """
        propagate the mask through the video
        """
        frame_iterator = DataLoader(
            Video(self.img_dir, get_transforms()),
            batch_size=1, 
            shuffle=False, 
            pin_memory=True
        )

        total_m = (len(frame_iterator) - 1) // mem_freq + 2 # +1 for first frame, +1 for zero start indexing

        propagation_model = TopKSTM(
            total_m, 
            model_device, 
            memory_device, 
            top_k, 
            mem_freq
        )
        propagation_model.load_pretrained(model_weights)
        propagation_model.eval()

        # get first frame in video
        frame = next(iter(frame_iterator))

        # get mask of initial frame
        mask = np.array(Image.open(initial_mask))
        mask = get_transforms()([mask])[0]
        mask = mask.unsqueeze(0)
        
        propagation_model.add_to_memory(frame, mask, extend_memory=True)
        mask, _ = pad_divide_by(mask, 16)
        mask = F.interpolate(mask, (self.frame_size[1], self.frame_size[0]), mode="bilinear")
        save_frame(mask, path.join(save_dir, f"00000.png"))

        # loop through video and propagate mask, skipping first frame
        for i, frame in enumerate(frame_iterator):
            print(f"Propagating Mask: {i} / {len(frame_iterator)-1}")

            if i == 0:
                continue

            # predict of frame based on memory and append features of current frame to memory
            mask_pred = propagation_model.predict_mask_and_memorize(i, frame)

            # resize to correct output size
            mask_pred = F.interpolate(mask_pred, (self.frame_size[1], self.frame_size[0]), mode="bilinear")

            # save mask as image
            save_frame(mask_pred, path.join(save_dir, f"{i:05}.png"))

    def __getitem__(self, idx: int) -> torch.Tensor:
        masks = []

        for i_object in range(self.N_objects):
            mask_path = path.join(self.mask_dir, f"{i_object:02}", f"{idx:05}.png")
            masks.append(cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), self.frame_size))

        masks = np.float32(np.stack(masks))
        masks = torch.from_numpy(masks)

        binary_masks = (masks > 0.5).float()
        masks = masks * 2 - 1
        
        trimaps = self.mask2trimap(masks)

        trimaps = trimaps.unsqueeze(1).to(self.device)
        binary_masks = binary_masks.unsqueeze(1).to(self.device)

        return trimaps, binary_masks

    def __len__(self):
        return len(listdir(path.join(self.mask_dir, "00"))) - 1

    def mask2trimap(self, mask: torch.Tensor, trimap_width: int = 20):
        """Convert binary mask to trimap with values in [-1, 0, 1]."""
        fg_mask = (mask > 0).float()
        bg_mask = (mask < 0).float()
        trimap_width *= bg_mask.shape[-1] / self.frame_size[0]
        trimap_width = int(trimap_width)
        bg_mask = cv2.erode(bg_mask.numpy(), kernel=np.ones((trimap_width, trimap_width)), iterations=1)
        bg_mask = torch.from_numpy(bg_mask)
        mask = fg_mask - bg_mask
        return mask
