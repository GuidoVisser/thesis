import os
import numpy as np
import cv2
import json
from PIL import Image
from moviepy.editor import ImageSequenceClip, CompositeVideoClip, concatenate_videoclips

from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from utils.transforms import DeNormalize


def load_frame(filepath, ismask=False):
    if ismask:
        return to_tensor(np.array(Image.open(filepath)).astype(np.uint8))
    else:
        return to_tensor(np.array(Image.open(filepath)).astype(np.uint8)).unsqueeze(0)


def save_frame(frame, filepath, ismask=False, denormalize=False):

    if denormalize:
        frame = DeNormalize()([frame])

    save_image(frame[0], filepath)


def opencv_folder_to_video(dir_path, save_path, fps=24):
    image_files = [dir_path+'/'+img for img in sorted(os.listdir(dir_path)) if img.endswith(".png")] + \
                  [dir_path+'/'+img for img in sorted(os.listdir(dir_path)) if img.endswith(".jpg")]

    img = cv2.imread(image_files[0])

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))

    for img_path in image_files:
        print(img_path)
        img = cv2.imread(img_path)
        out.write(img)
    out.release()


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


def video_to_folder(video, dir_path, image_ext=".png", start_frame=0):
    """
    Generate a folder of images in sequence based on the frames of a video and save the fps for reconstruction

    Args:
        video (str): path to the video that is to be processed
        dir_path (str): path to the directory where the image sequence will be saved
        image_ext (str): extension of the images
        start_frame (int): index of first frame to be saved
    """
    # Load video
    vidcap = cv2.VideoCapture(os.path.join(dir_path, video))

    # create a save directory
    save_path = os.path.join(dir_path, video)
    save_path = save_path.replace(" ", "_")
    if save_path[-4:] == ".mp4":
        save_path = save_path[:-4] + "_480p"

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # save video info
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    with open(os.path.join(save_path, "info.json"), "w") as f:
        json.dump({"fps": fps}, f)

    # save frames
    # success, frame = vidcap.read()
    success = True
    count = 0
    # assert success, f"Could not read first frame of {video}"
    while success:
        success, frame = vidcap.read()
        
        if not success:
            break

        if count % 2 == (start_frame % 2) and count >= start_frame:
            frame = cv2.resize(frame, (720, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(save_path, f"{(count - start_frame) // 2:05d}{image_ext}"), frame)
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
    frame_files = [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder)) if frame.endswith(".png")] + \
                  [os.path.join(video_folder, frame) for frame in sorted(os.listdir(video_folder)) if frame.endswith(".jpg")]
    mask_files = [os.path.join(mask_folder, frame) for frame in sorted(os.listdir(mask_folder)) if frame.endswith(".png")] +\
                 [os.path.join(mask_folder, frame) for frame in sorted(os.listdir(mask_folder)) if frame.endswith(".jpg")]

    assert len(frame_files) == len(mask_files), f"Expected video and masks to have same amount of frames. Got {len(frame_files)} and {len(mask_files)}"

    clip = ImageSequenceClip(frame_files, fps=fps)    
    mask = ImageSequenceClip(mask_files, fps=fps)
    mask = mask.set_opacity(mask_opacity)

    masked_clip = CompositeVideoClip([clip, mask])
    if save_path is not None:
        masked_clip.write_videofile(save_path, fps=fps)

    return masked_clip


def create_masked_completion_demo(video_dir, mask_dir, completed_dir, save_path=None, fps=24):
    original = folder_to_video(video_dir, fps=fps)
    masked = create_masked_video(video_dir, mask_dir, fps=fps)
    completed = folder_to_video(completed_dir, fps=fps)

    final = concatenate_videoclips([original, masked, completed])

    if save_path is not None:
        final.write_gif(save_path, fps=fps)
    
    return final