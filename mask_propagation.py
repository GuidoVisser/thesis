import numpy as np
import torch
import glob
from tqdm import tqdm
from os import path, makedirs
from PIL import Image
from skimage.transform import warp

from utils import create_dir
from define_args import DEFAULT_ARGS
from visualisation import create_masked_video

from FGVC.RAFT.utils.flow_viz import flow_to_image
from FGVC.RAFT import RAFT
from FGVC.RAFT import utils as RAFT_utils


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


def predict_next_mask(current_mask, flow):
    """
    Predict the mask of the next frame by warping it in accordance with the flow

    Inputs:
        current_mask (np.array): the mask on the current frame
        flow(np.array): the forward flow from the current frame to the next frame
    
    Returns:
        predicted_mask (np.array): the predicted mask on the next frame
    """
    pass
    


def propagate_mask(args, mask, flow):

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
    args.outroot = "results/mask_propagation/DAVIS/480p"
    args.mask_dir = "datasets/DAVIS/Annotations/480p"
    args.video = "tennis"
    
    main(args)
    