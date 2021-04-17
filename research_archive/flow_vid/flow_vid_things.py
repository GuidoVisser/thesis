from os import path, listdir, mkdir
from shutil import copyfile
import cv2

from visualisation import folder_to_video

# full_vid = sorted(listdir("datasets/flow_vid/last300_Guido_ikea_720p"))
# print(full_vid)
# separations = 5


# mkdir("results/flow_vid_separated")
# for i in range(separations):
#     mkdir(f"results/flow_vid_separated/{i}")

# for frame in range(len(full_vid)):
#     copyfile(f"datasets/flow_vid/last300_Guido_ikea_720p/{full_vid[frame]}", f"results/flow_vid_separated/{frame % separations}/{full_vid[frame]}")

root_dir = "results/flow_vid_separated/completed/"
vids = sorted(listdir(root_dir))
target_dir = path.join(root_dir, "full_vid")
if not path.exists(target_dir):
    mkdir(target_dir)

for i in range(len(vids) - 1):
    vid = path.join(root_dir, vids[i]) 
    for j, frame in enumerate(sorted(listdir(vid))):
        fr_name = f"{(4400 + j*5 + i%5):5d}.png"
        copyfile(path.join(vid, frame), path.join(target_dir, fr_name))

folder_to_video(target_dir, "final.mp4", fps=25)