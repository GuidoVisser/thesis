import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from os import path, listdir
from typing import Union

from datasets import Video
from utils.utils import create_dir, create_dirs
from utils.video_utils import save_frame
from models.TopkSTM.utils import pad_divide_by
from models.TopkSTM.topkSTM import TopKSTM
from InputProcessing.utils.transforms import ToTensor, Compose, Normalize

class MaskHandler(object):
    def __init__(self,
                 args) -> None:
        super().__init__()

        self.device  = args.device
        self.root    = path.join(args.out_dir, "masks")
        self.img_dir = args.img_dir
        self.foreground_dir = path.join(args.out_dir, "foreground_masks")
        create_dirs(self.foreground_dir, self.root)
        
        # set hyperparameters
        self.frame_size = (args.frame_width, args.frame_height)
        self.N_objects  = len(listdir(args.mask_dir))

        # propagate each object mask through video
        for i in range(self.N_objects):
            input_dir = path.join(args.mask_dir, f"{i:02}")
            save_dir  = path.join(self.root, f"{i:02}")
            if not path.exists(save_dir):
                create_dir(save_dir)

                # if the path points to a single mask, propagate it through the video
                if len(listdir(input_dir)) == 1:
                    mask_path = path.join(input_dir, listdir(input_dir)[0])
                    self.propagate(mask_path, 50, 10, self.device, self.device, args.propagation_model, save_dir)
                    self.propagate(mask_path, 50, 10, self.device, self.device, args.propagation_model, save_dir, forward=False)

                # if the path points to a directory with masks, resize and copy them
                else:
                    for frame in sorted(listdir(input_dir)):
                        
                        # read
                        fn = path.join(input_dir, frame)
                        img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
                        
                        # resize
                        img = cv2.resize(img, self.frame_size)
                        if len(img.shape) == 2:
                            img = np.expand_dims(img, axis=2)
                        
                        # save
                        cv2.imwrite(path.join(save_dir, frame), img)
        
        self.prepare_foreground_masks()        

    @torch.no_grad()
    def propagate(self, initial_mask_path, top_k, mem_freq, model_device, memory_device, model_weights, save_dir, forward=True):
        """
        propagate the mask through the video
        """

        data_transforms = Compose([ToTensor(), Normalize()])

        dataset = Video(self.img_dir, data_transforms, frame_size=self.frame_size, forward=forward)
        length_video = len(dataset)
        
        initial_index = int((path.splitext(initial_mask_path)[0]).split("/")[-1])
        
        if forward:
            dataset = Subset(dataset, range(initial_index, length_video))
        else:
            dataset = Subset(dataset, range(length_video - initial_index - 1, length_video))

        frame_iterator = DataLoader(
            dataset,
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

        # get the frame in video for initial mask
        frame = next(iter(frame_iterator))

        # get mask of initial frame
        mask = cv2.imread(initial_mask_path, cv2.IMREAD_GRAYSCALE)
        mask = ToTensor()([mask])[0]
        mask = mask.unsqueeze(0)
        
        mask, _ = pad_divide_by(mask, 16)
        mask = F.interpolate(mask, (self.frame_size[1], self.frame_size[0]), mode="bilinear")
        propagation_model.add_to_memory(frame, mask, extend_memory=True)
        save_frame(mask, path.join(save_dir, f"{initial_index:05}.png"))

        # loop through video and propagate mask, skipping first frame
        for i, frame in enumerate(frame_iterator):
            # if forward:
            #     print(f"Propagating Mask: {i + initial_index} / {length_video - 1}")
            # else:
            #     print(f"Propagating Mask: {initial_index - i} / {length_video - 1}")

            if i == 0:
                continue

            # predict of frame based on memory and append features of current frame to memory
            mask_pred = propagation_model.predict_mask_and_memorize(i, frame)

            # resize to correct output size
            mask_pred = F.interpolate(mask_pred, (self.frame_size[1], self.frame_size[0]), mode="bilinear")

            # save mask as image
            if forward:
                save_frame(mask_pred, path.join(save_dir, f"{i + initial_index:05}.png"))
            else:
                save_frame(mask_pred, path.join(save_dir, f"{initial_index - i:05}.png"))

        del propagation_model

    def prepare_foreground_masks(self):
        """
        Prepare a directory with all masks of all foreground object for every frame
        """

        for frame in range(len(self)):
            foreground_masks = []
            for layer in sorted(listdir(self.root)):
                mask = cv2.imread(path.join(self.root, layer, f"{frame:05}.png"))
                foreground_masks.append((mask >= 128).astype('uint8'))
            
            foreground_mask = np.minimum(np.sum(np.stack(foreground_masks), axis=0), np.ones_like(foreground_masks[0])) * 255

            foreground_mask = cv2.dilate(foreground_mask.astype('uint8'), kernel=np.ones((3,3)))

            cv2.imwrite(path.join(self.foreground_dir, f"{frame:05}.png"), foreground_mask)

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        
        masks = []
        for i_object in range(self.N_objects):
            if isinstance(idx, slice):
                mask_paths = [path.join(self.root, f"{i_object:02}", f"{frame_idx:05}.png") for frame_idx in range(idx.start or 0, idx.stop or len(self), idx.step or 1)]
                object_masks = []
                for mask_path in mask_paths:
                    object_masks.append(cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), self.frame_size))
                masks.append(np.stack(object_masks))
            else:
                mask_path = path.join(self.root, f"{i_object:02}", f"{idx:05}.png")
                masks.append(cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), self.frame_size))

        masks = np.float32(np.stack(masks)) / 255.
        masks = torch.from_numpy(masks)

        binary_masks = (masks > 0.5).float()
        masks = masks * 2 - 1
        
        if len(masks.shape) == 4:
            trimaps = []
            for i in range(masks.shape[1]):
                trimaps.append(self.mask2trimap(masks[:, i]))
            trimaps = torch.stack(trimaps, dim=1)
        else:
            trimaps = self.mask2trimap(masks)

        trimaps = trimaps.unsqueeze(1)
        binary_masks = binary_masks.unsqueeze(1)

        return trimaps, binary_masks

    def __len__(self):
        return len(listdir(path.join(self.root, "00")))

    def mask2trimap(self, mask: torch.Tensor, trimap_width: int = 20):
        """Convert binary mask to trimap with values in [-1, 0, 1]."""
        fg_mask = (mask > 0).float()
        bg_mask = (mask < 0).float()[0]
        trimap_width *= bg_mask.shape[-1] / self.frame_size[0]
        trimap_width = int(trimap_width)
        bg_mask = cv2.erode(bg_mask.numpy(), kernel=np.ones((trimap_width, trimap_width)), iterations=1)
        bg_mask = torch.from_numpy(bg_mask).unsqueeze(0)
        mask = fg_mask - bg_mask
        return mask
