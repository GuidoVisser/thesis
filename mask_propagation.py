# python built-ins
import glob
from os import path, makedirs, listdir

# libraries
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.transform import warp
from torchvision.utils import save_image

# local modules
from utils.utils import create_dir
from utils.video_utils import create_masked_video, load_frame, save_frame
from utils.flow_utils import normalize_optical_flow, load_flow_frame
from models.MaskPropagationVAE import MaskPropVAE

# RAFT
from models.RAFT.utils.flow_viz import flow_to_image
from models.RAFT import RAFT
from models.RAFT import utils as RAFT_utils


@torch.no_grad()
def propagate_mask_through_video(model, video_dir, flow_dir, out_dir, initial_mask):
    # get filepaths of data
    frames = [path.join(video_dir, frame) for frame in sorted(listdir(video_dir))]
    flow_frames = [path.join(flow_dir, flow_frame) for flow_frame in sorted(listdir(flow_dir))]

    # make sure mask has the right amount of dimensions
    mask = load_frame(initial_mask, ismask=True)
    for i in range(4 - len(mask.size())):
        mask = mask.unsqueeze(0)

    # initialize padder
    padder = RAFT_utils.utils.InputPadder(mask.size())
    mask = padder.pad(mask)[0]
    mask = mask.to(model.device)

    # save initial mask
    create_dir(out_dir)
    save_frame(mask, path.join(out_dir, "00000.png"), ismask=True)

    for i in range(len(frames)-1):
        # load data
        frame = load_frame(frames[i+1])
        flow_frame = load_flow_frame(flow_frames[i])

        # pad data to match model dimensions
        frame = padder.pad(frame)[0]

        # set device of data to match the model
        frame = frame.to(model.device)
        flow_frame = flow_frame.to(model.device)

        flow_frame = normalize_optical_flow(flow_frame)

        # predict next mask
        next_mask = model.predict_next_mask(mask, flow_frame, frame)
        
        # save next mask
        save_frame(next_mask, path.join(out_dir, f"{i+1:05d}.png"), ismask=True)

        # Use next mask as current for next iteration
        mask = next_mask
    

def propagate_mask(args):
    model = VAE(num_filters=32, z_dim=200)
    model.load_state_dict(torch.load(args.mask_prop_model))
    model.to('cuda')
    model.eval()

    propagate_mask_through_video(model, args.video_dir, args.flow_dir, args.mask_dir, args.initial_mask)
    

def calculate_flow(args, model, video, mode):
    """
    Calculates either the forward or backward optical flow for the given video

    Args:
        args (Namespace): Namespace object containing settings for the code
        model (torch.module): model for estimating the optical flow of the video
        video (torch.Tensor): video
        mode (str['forward', 'backward']): option for forward or backward flow

    Returns:
        Flow (np.ndarray): array containing the flow of the entire video
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    if path.isdir(path.join(args.outroot, args.video, 'flow', mode + '_flo')):
        for flow_name in sorted(glob.glob(path.join(args.outroot, args.video, 'flow', mode + '_flo', '*.flo'))):
            print("Loading {0}".format(flow_name), '\r', end='')
            flow = RAFT_utils.frame_utils.readFlow(flow_name)
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)
        return Flow

    create_dir(path.join(args.outroot, args.video, 'flow', mode + '_flo'))
    create_dir(path.join(args.outroot, args.video, 'flow', mode + '_png'))

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            # Flow visualization.
            flow_img = flow_to_image(flow)
            flow_img = Image.fromarray(flow_img)

            # Saves the flow and flow_img.
            flow_img.save(path.join(args.outroot, args.video, 'flow', mode + '_png', '%05d.png'%i))
            RAFT_utils.frame_utils.writeFlow(path.join(args.outroot, args.video, 'flow', mode + '_flo', '%05d.flo'%i), flow)

    return Flow
   


def propagate_mask_naive(args, mask, flow):

    frame_height, frame_width = mask.shape

    # remove padding from RAFT model
    flow = flow[:frame_height, :frame_width, :, :]

    n_frames = flow.shape[3]
    row_coords, col_coords = np.meshgrid(np.arange(frame_height), np.arange(frame_width), indexing='ij')

    print("Propagating initial mask with flow")
    propagated_masks = [mask]
    for frame in tqdm(range(n_frames)):
        previous_mask = propagated_masks[-1]
        new_mask = np.zeros(mask.shape)

        pixel_index_row = row_coords + np.round(flow[:, :, 1, frame].astype(int))
        pixel_index_col = col_coords + np.round(flow[:, :, 0, frame].astype(int))

        pixel_index_row = np.maximum(np.minimum(pixel_index_row, frame_height-1), 0)
        pixel_index_col = np.maximum(np.minimum(pixel_index_col, frame_width-1), 0)

        for x in range(frame_width):
            for y in range(frame_height):
                new_mask[pixel_index_row[y, x], pixel_index_col[y, x]] += previous_mask[y, x]

        propagated_masks.append(new_mask)

        im = Image.fromarray(new_mask).convert("RGB")
        im.save(path.join(args.outroot, args.video, f"masks/{frame+1:05d}.png"))

    propagated_masks = np.stack(propagated_masks, axis=-1)

    return propagated_masks

def initialize_RAFT(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.vc_model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def main(args):

    # define model
    RAFT_model = initialize_RAFT(args)

    # load video
    filename_list = glob.glob(path.join(args.data_dir, args.video, '*.png')) + \
                    glob.glob(path.join(args.data_dir, args.video, '*.jpg'))

    video = [torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)).permute(2, 0, 1).float() for filename in sorted(filename_list)]
    video = torch.stack(video, dim=0).to('cuda')

    # calculate flow
    flow = calculate_flow(args, RAFT_model, video, mode="forward")

    initial_mask = np.array(Image.open(path.join(args.mask_dir, args.video, "00000.png"))).astype(np.uint8)
    
    im = Image.fromarray(initial_mask).convert("RGB")
    im.save(path.join(args.outroot, args.video, "masks/00000.png"))

    propagated_mask = propagate_mask(args, initial_mask, flow)

    create_masked_video(
        path.join(args.data_dir, args.video), 
        path.join(args.outroot, args.video, "masks"),
        path.join(args.outroot, args.video, "demo.gif"),
        fps=15,
        mask_opacity=0.6 
    )


if __name__ == "__main__":
    
    args = DEFAULT_ARGS
    args.mask_dir = "results/mask_propagation/DAVIS/480p/bear/masks"
    args.video_dir = "datasets/DAVIS/JPEGImages/480p/bear"
    args.initial_mask = "datasets/DAVIS/Annotations/480p/bear/00000.png"
    args.flow_dir = "datasets/DAVIS/Flow/480p/flo/forward/bear"
    args.mask_prop_model = "results/VAE_logs/2021_04_14_18_40_25/checkpoints/epoch.pt"
    
    propagate_mask(args)
    