from os import rename, path, listdir
import cv2
from utils.utils import create_dir
import numpy as np

def correct_mivos_frame_bug(mask_dir, error_frame):

    masks = [path.join(mask_dir, fn) for fn in sorted(listdir(mask_dir))]
    last_frame = len(masks) - 1
    
    rename(masks[error_frame], path.join(mask_dir, "buffer.png"))

    for i in range(error_frame, len(masks) - 1):
        rename(masks[i+1], masks[i])

    rename(path.join(mask_dir, "buffer.png"), path.join(mask_dir, f"{last_frame:05}.png"))
        

def remove_first_n_frames(frame_dir, mask_dir, start_frame):

    # frames = [path.join(frame_dir, fn) for fn in sorted(listdir(frame_dir))]
    masks = [path.join(mask_dir, fn) for fn in sorted(listdir(mask_dir))]

    for i in range(len(masks) - start_frame):

        rename(masks[i+start_frame], masks[i])
        # rename(frames[i+start_frame], frames[i])

def mivos_to_masks(mask_dir, n_objects=1):

    for o in range(n_objects):
        create_dir(path.join(mask_dir, f"../{o+1:02}"))

    mask_paths = [path.join(mask_dir, fn) for fn in sorted(listdir(mask_dir))]

    for i, mask_path in enumerate(mask_paths):
        mask = cv2.imread(mask_path) / 128 * 255

        if n_objects >= 1:
            img = np.expand_dims(mask[:, :, 2], axis=2)
            img = np.repeat(img, 3, axis=2)

            cv2.imwrite(path.join(mask_dir, "..", "01", f"{i:05}.png"), img)
        
        if n_objects >= 2:
            img = np.expand_dims(mask[:, :, 1], axis=2)
            img = np.repeat(img, 3, axis=2)

            cv2.imwrite(path.join(mask_dir, "..", "02", f"{i:05}.png"), img)
            


if __name__ == "__main__":
    # vids = ["VID_20211211_130515.mp4", "VID_20211211_134633.mp4", "VID_20211211_134645.mp4", "VID_20211211_140403.mp4"]
    # dir_path = "datasets/Jaap_Jelle"
    # start_frames = [100, 0, 5, 0]

    # for i in range(len(vids)):
    #     video_to_folder(vids[i], dir_path, start_frame=start_frames[i])

    mivos_to_masks("datasets/Jaap_Jelle/Annotations/ringdijk/mask", 1)

    remove_first_n_frames("", "datasets/Jaap_Jelle/JPEGImages/480p/amsterdamse_brug", 6)