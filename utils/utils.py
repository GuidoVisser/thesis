import os
import random
import torch
import numpy as np

from datetime import datetime

def create_dir(dir_path):
    """
    Creates a directory if not exist.
    """
    os.makedirs(dir_path, exist_ok=True)

def create_dirs(dir_paths):
    """
    Creates a directory for all given paths
    """
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)

def collate_fn(batch):
    return tuple(zip(*batch))

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def time_function(func, *args, **kwargs):
    
    def wrapper(func, *args, **kwargs):
        t = datetime.now()
        output = func(*args, **kwargs)
        t_diff = datetime.now() - t
        return output

    return wrapper(func, *args, **kwargs)

def generate_image_pairs(video: str, outfile: str) -> None:
    """
    Generate a .txt file containing the file paths of subsequent image pairs in a video

    Args:
        video (str): path to the video directory
        outfile (str): path to the output .txt file
    """

    frame_list = sorted(os.listdir(video))
    # frame_list = [os.path.join(video, frame) for frame in frame_list]

    # check if directory of outfile exists
    outdir = "/".join(outfile.split("/")[:-1])
    if not os.path.exists(outdir):
        create_dir(outdir)

    with open(outfile, "a") as f:
        for i in range(len(frame_list) - 1):
            f.write(f"{frame_list[i]} {frame_list[i+1]}\n")

    return

# TODO implement padder
class Padder(object):
    def __init__(self, divide_by: int) -> None:
        super().__init__()

if __name__ == "__main__":
    DAVIS_vids = os.listdir("datasets/DAVIS/JPEGImages/480p")
    for vid in DAVIS_vids:
        generate_image_pairs(os.path.join("datasets/DAVIS/JPEGImages/480p", vid), f"datasets/DAVIS/SubsequentFrames/{vid}.txt")
        