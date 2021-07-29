from torch.nn.modules.container import T
from utils.utils import create_dirs
from InputProcessing.inputProcessor import InputProcessor
import cv2
import numpy as np
from os import listdir
from datetime import datetime

if __name__ == "__main__":
    
    video           = "tennis"
    results_dir     = "results/layer_decomposition"
    T               = datetime.now()

    img_dir         = f"datasets/DAVIS_minisample/JPEGImages/480p/{video}"
    initial_mask    = f"datasets/DAVIS_minisample/Annotations/480p/{video}/00000.png"
    mask_dir        = f"{results_dir}/{T}/masks/{video}"
    flow_dir        = f"{results_dir}/{T}/flow/{video}"
    background_dir  = f"{results_dir}/{T}/background/{video}"
    demo_dir        = f"{results_dir}/{T}"
    create_dirs(mask_dir, flow_dir, background_dir)
    
    ip = InputProcessor(img_dir, mask_dir, initial_mask, flow_dir, background_dir)
    img, flow_img, mask, matte, flow_matte_img, flow_conf, noise = ip[5]