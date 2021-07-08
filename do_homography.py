from BackgroundAttentionVolume.backgroundVolume import BackgroundVolume
from utils.video_utils import opencv_folder_to_video

import torch
import cv2
import numpy as np

if __name__ == "__main__":

    davis_video = "dance-jump"
    dataset = "DAVIS"
    frame = 20
    save_dir = f"datasets/{dataset}/BackgroundVolumes/480p/{davis_video}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    background_volume = BackgroundVolume(f"datasets/{dataset}/JPEGImages/480p/{davis_video}",
                                        f"datasets/{dataset}/topkSTMAnnotations/480p/{davis_video}", 
                                        device, 
                                        save_dir=save_dir,
                                        interval=1)

    # aligned_frames = background_volume.construct_full_volume()

    save_dir = f"datasets/{dataset}/BackgroundVolumes/480p/{davis_video}/bg_demo"
    for frame in [0,10,20]:

        aligned_frames = background_volume.construct_frame_volume(frame)

        left_img = cv2.imread(f"datasets/{dataset}/JPEGImages/480p/{davis_video}/{frame:05}.jpg")

        for i, right_img in enumerate(aligned_frames):
            img = np.concatenate((left_img, right_img), axis=1)
            img = img.astype(np.uint8)

            cv2.imwrite(f"{save_dir}/{frame*len(aligned_frames)+i:05}.jpg", img)

    opencv_folder_to_video(save_dir, 
                        f"{save_dir}/demo.mp4", fps=12)
