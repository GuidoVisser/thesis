import argparse
import numpy as np

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
parser.add_argument('--video', default='tennis', help='the name of the video')
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


######################
###    get args   ####
######################
DEFAULT_ARGS = parser.parse_args()

