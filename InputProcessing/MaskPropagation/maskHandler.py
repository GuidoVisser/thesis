import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from PIL import Image
from os import path

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
                 frame_size: list,
                 binary_threshold: float = 0.9) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        create_dir(self.mask_dir)

        self.size = frame_size

        self.binary_threshold = binary_threshold
         
    @torch.no_grad()
    def propagate(self, initial_mask, top_k, mem_freq, model_device, memory_device, model_weights):
        """
        propagate the mask through the video
        """
        frame_iterator = DataLoader(
            Video(self.img_dir, get_transforms()),
            batch_size=1, 
            shuffle=False, 
            pin_memory=True
        )

        total_m = (len(frame_iterator) - 1) // self.mem_freq + 2 # +1 for first frame, +1 for zero start indexing

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
        save_frame(mask, path.join(self.mask_dir, f"00000.png"))

        # loop through video and propagate mask, skipping first frame
        for i, frame in enumerate(self.frame_iterator):

            if i == 0:
                continue

            # predict of frame based on memory and append features of current frame to memory
            mask_pred = propagation_model.predict_mask_and_memorize(i, frame)

            # save mask as image
            save_frame(mask_pred, path.join(self.mask_dir, f"{i:05}.png"))

    def get_binary_mask(self, idx):
        """
        Get a binary mask for the frame with index `idx`
        """

        mask_path = path.join(self.mask_dir, f"{idx:05}.png")
        mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), self.size)
        _, mask = cv2.threshold(mask, self.binary_threshold, 1, cv2.THRESH_BINARY)

        mask = np.minimum(mask, np.ones(mask.shape))

        return mask