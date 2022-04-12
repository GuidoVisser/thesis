from compositing import Compositer
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


def new_composite_demo(root: str, fps=10, end_pause=3):

    decomposition_root = path.join(root, "decomposition/final")

    video_path = path.join(root, "final_demo.gif")
    vid_length = len(listdir(path.join(decomposition_root, "ground_truth")))
    num_object_layers = len(listdir(path.join(decomposition_root, "layers")))

    frames = []
    for frame_idx in range(vid_length):
        rgb_array   = []
        alpha_array = []
        flow_array  = []

        # Ground truth
        rgb_array.append(cv2.imread(path.join(decomposition_root, "ground_truth", f"{frame_idx:05}.png")))
        alpha_array.append(np.ones_like(rgb_array[0]) * 255)
        flow_array.append(cv2.imread(path.join(root, "flow/forward/png", f"{frame_idx:05}.png")))

        # Object layers
        for object_idx in range(1, num_object_layers):
            if object_idx == 1:
                rgb_array.append(cv2.imread(path.join(decomposition_root, "background", f"{frame_idx:05}.png")))

                alpha = cv2.imread(path.join(decomposition_root, "alpha", f"01/{frame_idx:05}.png"))
                alpha_array.append(alpha)

                flow_bg = cv2.imread(path.join(decomposition_root, "flow/png", f"00/{frame_idx:05}.png"))
                flow = cv2.imread(path.join(decomposition_root, "flow/png", f"01/{frame_idx:05}.png"))
                flow_array.append(((1 - alpha / 255) * flow_bg + alpha / 255 * flow).astype('uint8'))
            else:
                rgb_array.append(cv2.imread(path.join(decomposition_root, "layers", f"{object_idx:02}", f"{frame_idx:05}.png")))
                flow_array.append(cv2.imread(path.join(decomposition_root, "flow/png", f"{object_idx:02}", f"{frame_idx:05}.png")))
                alpha_array.append(cv2.imread(path.join(decomposition_root, "alpha", f"{object_idx:02}", f"{frame_idx:05}.png")))

        rgb_img   = np.concatenate(rgb_array, axis=1)
        alpha_img = np.concatenate(alpha_array, axis=1)
        flow_img  = np.concatenate(flow_array, axis=1)

        img = np.concatenate([alpha_img, rgb_img, flow_img], axis=0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        frames.append(img)

    frames.extend([img]*end_pause)

    frames = np.stack(frames)

    imageio.mimsave(video_path, frames, format="GIF", fps=fps)

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

def video_completion_demo(roots: dict, out_path: str, layers: list, fps: int = 10, end_pause: int = 3):


    # Dynamatte
    compositer = Compositer(roots["dynamatte"])
    img_array_dyn = []
    for i in range(len(listdir(roots["ground_truth"]))):
        img = compositer.composite_frame(i, layers)
        img = cv2.cvtColor(img[..., :3].astype(np.float32), cv2.COLOR_RGB2BGR) * 255.
        img_array_dyn.append(img.astype(np.uint8))
    img_array_dyn.extend([img]*end_pause)
    dynamatte = np.stack(img_array_dyn)

    # Omnimatte
    compositer = Compositer(roots["omnimatte"])
    img_array_om = []
    for i in range(len(listdir(roots["ground_truth"]))):
        img = compositer.composite_frame(i, layers)
        img = cv2.cvtColor(img[..., :3].astype(np.float32), cv2.COLOR_RGB2BGR) * 255.
        img_array_om.append(img.astype(np.uint8))
    img_array_om.extend([img]*end_pause)
    omnimatte = np.stack(img_array_om)

    h, w, c = img_array_dyn[0].shape

    # ground truth
    img_array_gt = []
    for fn in sorted(listdir(roots["ground_truth"])):
        img = cv2.resize(cv2.imread(path.join(roots["ground_truth"], fn)), (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_array_gt.append(img.astype(np.uint8))
    img_array_gt.extend([img]*end_pause)
    ground_truth = np.stack(img_array_gt)

    # FGVC
    img_array_fgvc = []
    for fn in sorted(listdir(roots["fgvc"])):
        if path.splitext(fn)[1] == ".mp4":
            continue

        img = cv2.resize(cv2.imread(path.join(roots["fgvc"], fn)), (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_array_fgvc.append(img.astype(np.uint8))
    img_array_fgvc.extend([img]*end_pause)
    fgvc = np.stack(img_array_fgvc[:-1])

    # Onion-Peel
    img_array_op = []
    for fn in sorted(listdir(roots["onion_peel"])):
        img = cv2.resize(cv2.imread(path.join(roots["onion_peel"], fn)), (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_array_op.append(img.astype(np.uint8))
    img_array_op.extend([img]*end_pause)
    onion_peel = np.stack(img_array_op[:-1])

    output_array = np.concatenate([ground_truth, dynamatte, omnimatte, fgvc, onion_peel], axis=2)

    imageio.mimsave(out_path, output_array, format="GIF", fps=fps)


if __name__ == "__main__":
    fps = 15
    end_pause = 5

    videos = {
        "kruispunt_rijks": [0, 1, 3],
        "ringdijk": [0, 1],
        "amsterdamse_brug": [0, 1, 3],
        "nescio_1": [0, 1],
        "nescio_2": [0, 1, 3],
        "cows": [0, 1],
        "hockey": [0, 1],
        "dance-jump": [0, 1],
        "car-roundabout": [0, 1],
        "scooter-black": [0, 1],
        "flamingo": [0, 1],
        "drift-chicane": [0, 1],
        "rollerblade": [0, 1]
    }
    for video, layers in videos.items():
        # roots = {
        #     "dynamatte":    f"results/final/dynamatte/{video}",
        #     "omnimatte":    f"results/final/omnimatte/{video}/decomposition/final",
        #     "ground_truth": f"results/final/omnimatte/{video}/decomposition/final/ground_truth",
        #     "fgvc":         f"results/final/fgvc/{video}/frame_seamless_comp_final",
        #     "onion_peel":   f"results/final/onion_peel/{video}"
        # }

        # video_completion_demo(roots, f"results/final/vc_{video}.gif", layers, fps, end_pause)

        fgvc_path = f"results/final/fgvc/{video}/frame_seamless_comp_final"
        for fn in listdir(fgvc_path):
            if path.splitext(fn)[1] == ".mp4":
                continue
            
            img = cv2.imread(path.join(fgvc_path, fn))
            img = cv2.resize(img, (448, 256))
            cv2.imwrite(path.join(fgvc_path, fn), img)

    # file_paths = [
    #     "results/layer_decomposition/scooter-black_10_20_18_24_28/decomposition/inference",
    #     "results/layer_decomposition_dynamic/scooter-black_10_28_19_02_45/decomposition/inference",
    #     "results/layer_decomposition_dynamic/scooter-black_10_29_04_18_35/decomposition/inference"
    # ]    

    # create_decomposite_demo(file_paths, fps, end_pause)

    # visualize_attention_map("results/layer_decomposition_dynamic/scooter-black/decomposition/final", channels=list(range(100)))