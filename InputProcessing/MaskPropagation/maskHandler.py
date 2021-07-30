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
                 binary_threshold: float = 0.7) -> None:
        super().__init__()

        # set hyperparameters
        self.size             = frame_size
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
        mask = F.interpolate(mask, (self.size[1], self.size[0]), mode="bilinear")
        save_frame(mask, path.join(save_dir, f"00000.png"))

        # loop through video and propagate mask, skipping first frame
        for i, frame in enumerate(frame_iterator):
            print(f"Propagating Mask: {i} / {len(frame_iterator)-1}")

            if i == 0:
                continue

            # predict of frame based on memory and append features of current frame to memory
            mask_pred = propagation_model.predict_mask_and_memorize(i, frame)

            # resize to correct output size
            mask_pred = F.interpolate(mask_pred, (self.size[1], self.size[0]), mode="bilinear")

            # save mask as image
            save_frame(mask_pred, path.join(save_dir, f"{i:05}.png"))

    def get_binary_masks(self, idx: int) -> list:
        """
        Get all binary masks for the frame with index `idx`
        """
        masks = []

        for i_object in range(self.N_objects):
            mask_path = path.join(self.mask_dir, f"{i_object:02}", f"{idx:05}.png")
            mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), self.size)
            _, mask = cv2.threshold(mask, self.binary_threshold*255, 1, cv2.THRESH_BINARY)

            masks.append(np.minimum(mask, np.ones(mask.shape)))

        return np.float32(np.stack(masks))

    def get_alpha_mask(self, idx):
        mask_path = path.join(self.mask_dir, f"{idx:05}.png")
        mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), self.size)
        return mask

    def __getitem__(self, idx: int) -> torch.Tensor:
        mask = torch.from_numpy(self.get_binary_masks(idx)).unsqueeze(1).to(self.device)
        return mask

    def __len__(self):
        return len(listdir(path.join(self.mask_dir, "00"))) - 1
