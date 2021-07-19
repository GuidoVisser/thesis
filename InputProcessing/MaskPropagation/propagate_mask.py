import torch
import argparse
from os import path
from datetime import datetime

from models.TopkSTM import TopKSTM
from models.TopkSTM.utils import pad_divide_by
from datasets import DAVISVideo
from utils.transforms import get_transforms
from utils.utils import create_dir
from utils.video_utils import create_masked_video, save_frame

@torch.no_grad()
def propagate_mask(video_dir: str,
                   output_dir: str,
                   initial_mask:str,
                   memory_frequency: int,
                   top_k: int,
                   model_device: str,
                   memory_device: str,
                   model_weights: str):

    # set up data iterable
    dataset = DAVISVideo(args.data_dir, args.video, get_transforms())
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True
    )

    # calculate necessary memory size
    total_m = (len(dataset) - 1) // args.mem_freq + 2 # +1 for first frame, +1 to make sure indexing remains within bounds

    # initialize model
    model = TopKSTM(
        total_m, 
        args.model_device, 
        args.memory_device, 
        args.top_k, 
        args.mem_freq
    )

    model.load_pretrained(args.model_path)
    model.eval()

    # set up memory with first frame and ground truth mask
    frame, mask = next(iter(dataloader))
    model.add_to_memory(frame, mask, extend_memory=True)
    mask, _ = pad_divide_by(mask, 16)
    frame, _ = pad_divide_by(frame, 16)
    save_frame(mask, path.join(mask_results_dir, f"00000.png"))
    save_frame(frame[0], path.join(frame_results_dir, f"00000.png"), denormalize=True)

    # loop through video and propagate mask, skipping first frame
    for i, (frame, _) in enumerate(dataloader):

        if i == 0:
            continue

        # predict of frame based on memory and append features of current frame to memory
        mask_pred = model.predict_mask_and_memorize(i, frame)

        # save mask as image
        save_frame(mask_pred, path.join(mask_results_dir, f"{i:05}.png"))
        frame, _ = pad_divide_by(frame, 16)
        save_frame(frame[0], path.join(frame_results_dir, f"{i:05}.png"), denormalize=True)

    # create video with propagated mask as overlay
    create_masked_video(frame_results_dir, mask_results_dir, save_path=path.join(results_dir, "demo.mp4"), mask_opacity=0.6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="datasets/DAVIS")
    parser.add_argument("--video", type=str, default="horsejump-high")
    parser.add_argument("--log_dir", type=str, default="results/topkSTM")
    parser.add_argument("--model_path", type=str, default="models/weights/MiVOS/propagation_model.pth")

    parser.add_argument("--mem_freq", type=int, default=10, help="Frequency at which to expand the memory")
    parser.add_argument("--top_k", type=int, default=50, help="top k channels of attention are used to reduce noise in the output")

    parser.add_argument("--model_device", type=str, default="cuda:0", help="specifies the device for the model")
    parser.add_argument("--memory_device", type=str, default="cuda:0", help="specifies the device for the memory")

    args = parser.parse_args()

    main(args)