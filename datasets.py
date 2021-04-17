import numpy as np
import torch
import pathlib
import random

from PIL import Image
from os import path, listdir


class DAVISPairsDataset(object):
    def __init__(self, frame_root, mask_root, transforms, device='cuda', max_distance=1):
        self.frame_root = frame_root
        self.mask_root = mask_root
        self.transforms = transforms
        self.device = device
        self.max_distance = max_distance
        self.frames = []
        for video in listdir(self.frame_root):
            self.frames.extend([path.join(video, frame) 
                                for frame 
                                in list(sorted(listdir(path.join(self.frame_root, video))))
            ])

    def __getitem__(self, idx):

        # Get path to frame and corresponding mask
        source_frame_path = path.join(self.frame_root, self.frames[idx])
        source_mask_path = path.join(self.mask_root, self.frames[idx])
        
        # get video title
        p = pathlib.Path(self.frames[idx])
        video = p.parts[0]

        # get index of current, first and last frame in directory
        idx_in_dir = int(p.parts[1][:-4])
        first_frame = 0
        last_frame = len(listdir(path.join(self.frame_root, video))) - 1

        # get the path of the other frame and corresponding mask
        candidates = list(range(
                        max(first_frame, idx_in_dir - self.max_distance), 
                        min(last_frame, idx_in_dir + self.max_distance) + 1
                    ))
        candidates.remove(idx_in_dir)
        candidates = [i - idx_in_dir for i in candidates]
        target_idx = idx + random.choice(candidates)
        
        target_frame_path = path.join(self.frame_root, self.frames[target_idx])
        target_mask_path = path.join(self.mask_root, self.frames[target_idx])

        # DAVIS images are .jpg but masks are .png; change extension accordingly
        source_mask_path = path.splitext(source_mask_path)[0] + ".png"
        target_mask_path = path.splitext(target_mask_path)[0] + ".png"

        # load images
        source_frame = np.array(Image.open(source_frame_path).convert("RGB"))
        target_frame = np.array(Image.open(target_frame_path).convert("RGB"))
        source_mask = np.array(Image.open(source_mask_path).convert("L"))
        target_mask = np.array(Image.open(target_mask_path).convert("L"))

        if self.transforms is not None:
            source_frame, source_mask = self.transforms(source_frame, source_mask)
            target_frame, target_mask = self.transforms(target_frame, target_mask)
        
        return source_frame, source_mask, target_frame, target_mask

    def __len__(self):
        return len(self.frames)


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(listdir(path.join(root, "PNGImages"))))
        self.masks = list(sorted(listdir(path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    

class TestVideoDataset(object):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(listdir(path.join(root, "tennis"))))
        self.masks = list(sorted(listdir(path.join(root, "tennis_mask"))))
        
    def __getitem__(self, idx):
        img_path = path.join(self.root, "tennis", self.imgs[idx])
        mask_path = path.join(self.root, "tennis_mask", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        target = Image.open(mask_path)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
