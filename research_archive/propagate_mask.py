import argparse
from torch.utils.data import DataLoader
from os import path
from datetime import datetime

from InputProcessing.MaskPropagation.maskHandler import MaskHandler
from datasets import Video
from utils.transforms import get_transforms


def main(args):

    iterator = Video(args.video_dir, get_transforms())
    dataloader = DataLoader(
        iterator, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True
    )

    mask_propagator = MaskHandler(
        dataloader, 
        args.initial_mask, 
        path.join(args.output_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S")),
        args.mem_freq, 
        args.top_k, 
        args.model_weights, 
        args.model_device, 
        args.memory_device)

    mask_propagator.propagate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir", type=str, default="datasets/DAVIS/JPEGImages/480p/horsejump-high")
    parser.add_argument("--initial_mask", type=str, default="datasets/DAVIS/Annotations/480p/horsejump-high/00000.png")
    parser.add_argument("--output_dir", type=str, default="results/topkSTM")
    parser.add_argument("--model_weights", type=str, default="models/weights/MiVOS/propagation_model.pth")

    parser.add_argument("--mem_freq", type=int, default=10, help="Frequency at which to expand the memory")
    parser.add_argument("--top_k", type=int, default=50, help="top k channels of attention are used to reduce noise in the output")

    parser.add_argument("--model_device", type=str, default="cuda:0", help="specifies the device for the model")
    parser.add_argument("--memory_device", type=str, default="cuda:0", help="specifies the device for the memory")

    args = parser.parse_args()

    main(args)