import torch
import argparse
from os import path, listdir
from datetime import datetime

from torchvision.utils import save_image

from models.TopkSTM import TopKSTM
from models.TopkSTM.utils import aggregate_wbg, pad_divide_by
from datasets import DAVISVideo
from utils.transforms import get_transforms
from utils.utils import create_dir
from utils.video_utils import create_masked_video

@torch.no_grad()
def main(args):

    # set up result directory
    results_dir = path.join(args.log_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    create_dir(results_dir)

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

    # loop through video and propagate mask, skipping first frame
    for i, (frame, _) in enumerate(dataloader, 1):

        # predict of frame based on memory and append features of current frame to memory
        mask_pred = model.predict_mask_and_memorize(i, frame)

        # save mask as image
        save_image(mask_pred, path.join(results_dir, f"{i:05}.png"))

    # create video with propagated mask as overlay
    create_masked_video(f"{args.data_dir}/JPEGImages/480p/{args.video}", results_dir, save_path=path.join(results_dir, "demo.mp4"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="datasets/DAVIS_sample")
    parser.add_argument("--video", type=str, default="tennis")
    parser.add_argument("--log_dir", type=str, default="results/topkSTM")
    parser.add_argument("--model_path", type=str, default="models/weights/MiVOS/propagation_model.pth")

    parser.add_argument("--mem_freq", type=int, default=10, help="Frequency at which to expand the memory")
    parser.add_argument("--top_k", type=int, default=50, help="top k channels of attention are used to reduce noise in the output")

    parser.add_argument("--model_device", type=str, default="cuda:0", help="specifies the device for the model")
    parser.add_argument("--memory_device", type=str, default="cuda:0", help="specifies the device for the memory")

    args = parser.parse_args()

    main(args)