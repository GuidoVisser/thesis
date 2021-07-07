from homography.construct_background_volume import construct_volume
from utils.video_utils import opencv_folder_to_video

import torch

if __name__ == "__main__":

    davis_video = "horsejump-high"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aligned_frames = construct_volume(f"datasets/DAVIS/JPEGImages/480p/{davis_video}",
                                      f"datasets/DAVIS/topkSTMAnnotations/480p/{davis_video}", 
                                      device, 
                                      save_dir=f"datasets/DAVIS/BackgroundVolumes/480p/{davis_video}")

    opencv_folder_to_video(f"datasets/DAVIS/BackgroundVolumes/480p/{davis_video}", 
                           f"datasets/DAVIS/BackgroundVolumes/480p/{davis_video}/demo.mp4")
