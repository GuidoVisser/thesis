import os
import numpy as np
import torch
import moviepy
import cv2
import json
from PIL import Image
from moviepy.editor import ImageSequenceClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip


def load_flow_frame(filepath):
    return torch.from_numpy(RAFT_utils.frame_utils.readFlow(filepath)).permute(2, 0, 1).unsqueeze(0)    

def load_frame(filepath, ismask=False):
    if ismask:
        return torch.from_numpy(np.array(Image.open(filepath)).astype(np.uint8)).float()
    else:
        return torch.from_numpy(np.array(Image.open(filepath)).astype(np.uint8)).permute(2, 0, 1).float().unsqueeze(0)

def save_frame(frame, filepath, ismask=False):
   
    if ismask:
        img = Image.fromarray(frame[0][0].cpu().detach().numpy()).convert('L')
    else:
        img = Image.fromarray(frame[0].cpu().permute(1, 2, 0).detach().numpy()).convert('RGB')
    
    img.save(filepath)

def folder_to_video(dir_path, save_path=None, fps=24):
    """
    generate a video from a folder containing images

    Args:
        dir_path (str): the path to the directory that is to be used for the video
        result_path (str): the path to where the video will be saved
        fps (int): frames per second of the video
    """
    image_files = [dir_path+'/'+img for img in sorted(os.listdir(dir_path)) if img.endswith(".png")] + \
                  [dir_path+'/'+img for img in sorted(os.listdir(dir_path)) if img.endswith(".jpg")]
    clip = ImageSequenceClip(image_files, fps=fps)

    if save_path is not None:
        if not save_path.endswith(".mp4"):
            raise ValueError("Please give an .mp4 save path")
        
        clip.write_videofile(save_path)

    return clip


def video_to_folder(video, dir_path, image_ext=".png"):
    """
    Generate a folder of images in sequence based on the frames of a video and save the fps for reconstruction

    Args:
        video (str): path to the video that is to be processed
        dir_path (str): path to the directory where the image sequence will be saved
        image_ext (str): extension of the images
    """
    # Load video
    vidcap = cv2.VideoCapture(os.path.join(dir_path, video))

    # create a save directory
    save_path = os.path.join(dir_path, video)
    save_path = save_path.replace(" ", "_")
    if save_path[-4:] == ".mp4":
        save_path = save_path[:-4] + "_720p"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # save video info
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump({"fps": fps}, f)

    # save frames
    success, frame = vidcap.read()
    count = 0
    assert success, f"Could not read first frame of {video}"
    while success:
        frame = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(save_path, f"{count:05d}{image_ext}"), frame)
        success, frame = vidcap.read()
        print(f"read new frame : {success}")
        count += 1
    
    return


def create_masked_video(video_folder, mask_folder, save_path=None, fps=24, mask_opacity=0.3):
    """
    Take the frames saved in {video_folder} and corresponding masks in {mask_folder} and
    creat a video with the mask overlayed over the frames.

    Args:
        video_folder (str): path to video
        mask_folder (str): path to masks
        save_path (str): path to save location of generated video
    """
    frame_files = [video_folder+'/'+frame for frame in sorted(os.listdir(video_folder)) if frame.endswith(".png")] + \
                  [video_folder+'/'+frame for frame in sorted(os.listdir(video_folder)) if frame.endswith(".jpg")]
    mask_files = [mask_folder+'/'+frame for frame in sorted(os.listdir(mask_folder)) if frame.endswith(".png")] +\
                 [mask_folder+'/'+frame for frame in sorted(os.listdir(mask_folder)) if frame.endswith(".jpg")]

    assert len(frame_files) == len(mask_files), f"Expected video and masks to have same amount of frames. Got {len(frame_files)} and {len(mask_files)}"

    clip = ImageSequenceClip(frame_files, fps=fps)    
    mask = ImageSequenceClip(mask_files, fps=fps)
    mask = mask.set_opacity(mask_opacity)

    masked_clip = CompositeVideoClip([clip, mask])
    if save_path is not None:
        masked_clip.write_gif(save_path, fps=fps)

    return masked_clip


def create_masked_completion_demo(video_dir, mask_dir, completed_dir, save_path=None, fps=24):
    original = folder_to_video(video_dir, fps=fps)
    masked = create_masked_video(video_dir, mask_dir, fps=fps)
    completed = folder_to_video(completed_dir, fps=fps)

    final = concatenate_videoclips([original, masked, completed])

    if save_path is not None:
        final.write_gif(save_path, fps=fps)
    
    return final


if __name__ == "__main__":
    print("Video utils: " \
        "\n\t- folder_to_video" \
        "\n\t- video_to_folder" \
        "\n\t- create_masked_video" \
        "\n\t- create_masked_completion_demo"
        "\n\t- load_frame" \
        "\n\t- save_frame" \
        "\n\t- load_flow_frame")
    
    # masked_video = create_masked_video(
    #     "datasets/DAVIS/JPEGImages/480p/lucia",
    #     "datasets/DAVIS/generated_masks/480p/lucia/combined",
    #     "results/VIS_FGVC/DAVIS/480plucia/lucia_masked.mp4"
    # )

    # video = "dog-agility"

    # demo_clip = create_masked_completion_demo(
    #     f"datasets/DAVIS/JPEGImages/480p/{video}",
    #     f"datasets/DAVIS/generated_masks/480p/{video}/combined",
    #     f"results/VIS_FGVC/DAVIS/480p{video}/frame_seamless_comp_final",
    #     f"results/VIS_FGVC/DAVIS/480p{video}/demo.gif"
    # )

    # dir_path = "datasets/flow_vid/"
    # vids = [vid for vid in os.listdir(dir_path) if vid[-4:] == ".mp4"]
    # for video in vids:
    #     video_to_folder(video, dir_path)
    # video_to_folder("full video Guido ikea lowres.mp4", dir_path)

    # folder_to_video("results/flow_vid/frame_comp_final", "results/flow_vid/completed.mp4", fps=25)