import numpy as np
import torch
import glob
from os import path, listdir
from PIL import Image
import argparse

from utils.utils import create_dir

from models.RAFT.utils.flow_viz import flow_to_image
from models.RAFT import RAFT
from models.RAFT import utils as RAFT_utils


def initialize_RAFT(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.RAFT_weights))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def calculate_batch_flow(flow_model, source_batch, target_batch):

    _, flow = flow_model(source_batch, target_batch, iters=20, test_mode=True)

    return flow


def save_flow(args, flow, mode, video, i):
    flow = flow[0].permute(1, 2, 0).cpu().numpy()
    
    # flow visualization.
    flow_img = RAFT_utils.flow_viz.flow_to_image(flow)
    flow_img = Image.fromarray(flow_img)

    # save flow and flow_image
    flow_img.save(path.join(args.data_root, "Flow/480p/png/", mode, video, '%05d.png'%i))
    RAFT_utils.frame_utils.writeFlow(path.join(args.data_root, "Flow/480p/flo/", mode, video, '%05d.flo'%i), flow)


def calculate_video_flow(args, flow_model, video, padder):

    frames = glob.glob(path.join(args.data_root, "JPEGImages/480p/", video, '*.jpg'))

    create_dir(path.join(args.data_root, "Flow/480p/png/forward/", video))
    create_dir(path.join(args.data_root, "Flow/480p/png/backward/", video))
    create_dir(path.join(args.data_root, "Flow/480p/flo/forward/", video))
    create_dir(path.join(args.data_root, "Flow/480p/flo/backward/", video))

    for i in range(len(frames) - 1):

        frame_1 = torch.unsqueeze(torch.from_numpy(np.array(Image.open(frames[i])).astype(np.uint8)).permute(2, 0, 1).float(), 0).to('cuda')
        frame_2 = torch.unsqueeze(torch.from_numpy(np.array(Image.open(frames[i+1])).astype(np.uint8)).permute(2, 0, 1).float(), 0).to('cuda')

        frame_1, frame_2 = padder.pad(frame_1, frame_2)

        flow = calculate_batch_flow(flow_model, frame_1, frame_2)
        save_flow(args, flow, "forward", video, i)
        flow = calculate_batch_flow(flow_model, frame_2, frame_1)
        save_flow(args, flow, "backward", video, i+1)

        print(i)

    return

@torch.no_grad()
def main(args):
    videos = listdir(path.join(args.data_root, "JPEGImages/480p"))
    
    flow_model = initialize_RAFT(args)
    padder = RAFT_utils.utils.InputPadder((1, 3, 480, 854))
    for video in videos:
        calculate_video_flow(args, flow_model, video, padder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', default = "datasets/DAVIS/", help="root directory of data")

    # RAFT
    parser.add_argument('--RAFT_weights', default='models/weights/zip_serialization_false/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    main(args)


