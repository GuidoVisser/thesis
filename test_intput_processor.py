from utils.utils import create_dirs
from InputProcessing.inputProcessor import InputProcessor
import cv2
import numpy as np
from os import listdir

if __name__ == "__main__":
    for video in listdir("datasets/DAVIS/JPEGImages/480p/"):
        print(video)
        img_dir =  f"datasets/DAVIS/JPEGImages/480p/{video}"
        mask_dir = f"results/topkSTM/{video}"
        initial_mask = f"datasets/DAVIS/Annotations/480p/{video}/00000.png"
        flow_dir = f"results/flow_testing/{video}"
        background_dir = f"results/background_testing/{video}"
        demo_dir = f"results/DAVIS_full_input/{video}"
        create_dirs(demo_dir, mask_dir)
        try:
            ip = InputProcessor(img_dir, mask_dir, initial_mask, flow_dir, background_dir)
            ip.input_demo(demo_dir)
        except:
            print("memory error, continueing")
            continue