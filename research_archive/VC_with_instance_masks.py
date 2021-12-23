from research_archive.models.third_party.VIS import MaskGenerator
from FGVC.tool.video_completion import *
from FGVC.RAFT.utils.utils import InputPadder
from os import path, listdir
import argparse
import mmcv
from PIL import Image
import glob
import torch
from torchvision import transforms
from torchvision.utils import save_image

from utils.video_utils import create_masked_completion_demo


def main(args):

    # pad frames in directory
    img_list = listdir(path.join(args.data_dir, args.video))

    img = mmcv.imread(path.join(args.data_dir, args.video, img_list[0]))
    padder = InputPadder([0,0,img.shape[0], img.shape[1]])

    frame_names = glob.glob(path.join(args.data_dir, args.video, "*.jpg")) +\
             glob.glob(path.join(args.data_dir, args.video, "*.png"))
    for frame_name in sorted(frame_names):

        try:
            frame = transforms.ToTensor()(Image.open(frame_name)).unsqueeze(0)
        except Exception:
            print(f"Error while processing the following image: {frame_name}")
        
        padded_frame = padder.pad(frame)[0].squeeze(0)
        save_image(padded_frame, frame_name)

    # generate masks
    cfg = mmcv.Config.fromfile(path.join(args.vis_cfg))
    MG = MaskGenerator(args.data_dir, args.mask_dir, cfg)
    # TODO: generate mask from ground truth segmentations
    MG.generate_masks_from_video_dir(args.video)
    # MG.combine_all_masks(args.video)

    # # perform video completion
    # args.outroot = path.join(args.outroot, args.video)

    # del MG

    video_completion(args)

    # create_masked_completion_demo(
    #     path.join(args.data_dir, args.video),
    #     path.join(args.mask_dir, args.video, "id_0"),
    #     path.join(args.outroot, "frame_seamless_comp_final"),
    #     path.join(args.outroot, "demo.gif")
    # )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    ######################
    ######   FGVC   ######
    ######################

    # video completion
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--data_dir', default='datasets/DAVIS/JPEGImages/480p/', help="dataset for evaluation")
    parser.add_argument('--mask_dir', default='datasets/DAVIS/generated_masks/480p/', help="mask for object removal")
    parser.add_argument('--video', default='lucia', help='the name of the video')
    parser.add_argument('--outroot', default='results/VIS_FGVC/DAVIS/480p/', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)

    # RAFT
    parser.add_argument('--vc_model', default='FGVC/weight/zip_serialization_false/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Deepfill
    parser.add_argument('--deepfill_model', default='FGVC/weight/zip_serialization_false/imagenet_deepfill.pth', help="restore checkpoint")

    # Edge completion
    parser.add_argument('--edge_completion_model', default='FGVC/weight/zip_serialization_false/edge_completion.pth', help="restore checkpoint")

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    # profiling
    parser.add_argument('--profile_path', default='profile.csv', help="path where profile data is saved")


    ######################
    ######   VIS    ######
    ######################

    parser.add_argument('--vis_cfg', default='MaskTrackRCNN/configs/masktrack_rcnn_r50_fpn_1x_youtubevos.py')

    args = parser.parse_args()

    main(args)



