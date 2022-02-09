import cv2
from os import path, listdir
import glob
import numpy as np
import imageio
from typing import Union
import torch
import torch.nn.functional as F
from utils.utils import create_dirs
from typing import Union
from math import ceil

def create_decomposite_demo(roots: Union[str, list], fps=10, end_pause=3, include_bg_layers=False):

    if isinstance(roots, str):
        video_path = path.join(roots, "demo.gif")
        roots = [roots]
    else:
        video_path = path.join(roots[0], "demo.gif")

    combined_array = []
    for root in roots:
        img_array = []
        for fn in sorted(glob.glob(path.join(root, "ground_truth/*.png"))):
            img = cv2.imread(fn)
            img_array.append(img)
        img_array.extend([img]*end_pause)

        for i, fn in enumerate(sorted(glob.glob(path.join(root, "background/*.png")))):
            img = cv2.imread(fn)
            img_array[i] = np.concatenate((img_array[i], img), axis=1)

        for j in range(end_pause):
            img_array[i+j+1] = np.concatenate((img_array[i+j+1], img), axis=1)

        object_imgs = []
        first_object = 0
        for object_idx, object in enumerate(sorted(listdir(path.join(root, "layers")))):

            if not include_bg_layers and object_idx <= 1:
                first_object += 1
                continue

            for i, fn in enumerate(sorted(glob.glob(path.join(root, "layers", object, "*.png")))):
                img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
                alpha = np.stack([img[:, :, 3]]*3, axis=2) / 255
                img = np.uint8(img[:, :, :3] * alpha)
                if object_idx == first_object:
                    object_imgs.append(img)
                else:
                    object_imgs[i] = np.concatenate((object_imgs[i], img), axis=1)

        object_imgs.extend([object_imgs[-1]] * end_pause)

        img_array   = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in img_array]
        object_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in object_imgs]

        img_array = np.stack(img_array)
        object_imgs_array = np.stack(object_imgs)
        
        img_array = np.concatenate((img_array, object_imgs_array), axis=2)

        combined_array.append(img_array)
    
    combined_array = np.concatenate(combined_array, axis=1)
    imageio.mimsave(video_path, combined_array, format="GIF", fps=fps)

def visualize_attention_map(root: str, channels: Union[list, None] = None, batch_width: int = 8, batch_height: int = 8, scale: int = 2):
    for frame_idx, img_path in enumerate(sorted(listdir(path.join(root, "ground_truth")))):
        img = cv2.imread(path.join(root, "ground_truth", img_path))
        attention_volume = torch.load(path.join(root, f"context_volumes/raw/{frame_idx:05}.pth"))

        H, W, _ = img.shape
        h, w = H//scale, W//scale

        if channels == None:
            _, num_channels, _, _ = attention_volume.shape
        else:
            num_channels = len(channels)

        assert num_channels > 0, "No valid input found"

        num_layers = attention_volume.shape[0]

        num_batches = ceil(num_channels / (batch_width*batch_height))
        batch_iterator = range(num_batches) if channels == None else channels

        for layer in range(num_layers):
            create_dirs(path.join(root, f"context_volumes/visualization/layer_{layer:02}"))
            create_dirs(*[path.join(root, f"context_volumes/visualization/layer_{layer:02}/{batch:02}") for batch in range(num_batches)])

        img = cv2.resize(img, (w, h))

        attention_volume = (attention_volume - torch.min(attention_volume)) / (torch.max(attention_volume) - torch.min(attention_volume)) * 255

        for layer in range(num_layers):
            channel_count = 0
            for batch in batch_iterator:

                imgs = []
                for i in range(batch_width):
                    img_row = []
                    for j in range(batch_height):

                        if channel_count < num_channels:
                            attn_map_idx = num_batches * batch + i * batch_width + j

                            attention_map = attention_volume[layer:layer+1, attn_map_idx:attn_map_idx+1]
                            attention_map = F.interpolate(attention_map, size=(h, w), mode='bilinear', align_corners=True)[0].permute(1, 2, 0).byte().numpy()
                            attention_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
                        else:
                            attention_map = np.zeros_like(attention_map)
                            img = np.zeros_like(img)

                        channel_count += 1

                        img_row.append(cv2.addWeighted(img, 0.7, attention_map, 0.3, 0))
                    imgs.append(np.concatenate(img_row, 1))
                imgs = np.concatenate(imgs, 0)

                cv2.imwrite(path.join(root, f"context_volumes/visualization/layer_{layer:02}/{batch:02}/{frame_idx:05}.png"), imgs)




if __name__ == "__main__":
    fps = 5
    end_pause = 5

    # file_paths = [
    #     "results/layer_decomposition/scooter-black_10_20_18_24_28/decomposition/inference",
    #     "results/layer_decomposition_dynamic/scooter-black_10_28_19_02_45/decomposition/inference",
    #     "results/layer_decomposition_dynamic/scooter-black_10_29_04_18_35/decomposition/inference"
    # ]    

    # create_decomposite_demo(file_paths, fps, end_pause)

    visualize_attention_map("results/layer_decomposition_dynamic/scooter-black/decomposition/final", channels=list(range(100)))