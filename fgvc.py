from models.third_party.FGVC.tool.video_completion import *
import argparse

def main(args):
    if args.seamless:
        video_completion_seamless(args)
    else:
        video_completion(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ######################
    ######   FGVC   ######
    ######################

    # video completion
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--data_dir', default='datasets/DAVIS/Images/', help="dataset for evaluation")
    parser.add_argument('--mask_dir', default='datasets/DAVIS/Annotations/', help="mask for object removal")
    parser.add_argument('--video', default='', help='the name of the video')
    parser.add_argument('--outroot', default='results/VIS_FGVC/DAVIS/480p/', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)

    # RAFT
    parser.add_argument('--RAFT_weights', default='models/third_party/weights/raft.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Deepfill
    parser.add_argument('--deepfill_model', default='research_archive/models/third_party/weights/imagenet_deepfill.pth', help="restore checkpoint")

    # Edge completion
    parser.add_argument('--edge_completion_model', default='research_archive/models/third_party/weights/edge_completion.pth', help="restore checkpoint")

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    # profiling
    parser.add_argument('--profile_path', default='profile.csv', help="path where profile data is saved")

    args = parser.parse_args()

    main(args)

