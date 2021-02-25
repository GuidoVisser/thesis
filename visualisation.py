import os
import moviepy
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def folder_to_video(dir_path, result_path, fps=24):
    """
    generate a video from a folder containing images

    Args:
        dir_path (str): the path to the directory that is to be used for the video
        result_path (str): the path to where the video will be saved
        fps (int): frames per second of the video
    """
    image_files = [dir_path+'/'+img for img in sorted(os.listdir(dir_path)) if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)

    if not result_path.endswith(".mp4"):
        result_path += ".mp4"
    
    clip.write_videofile(result_path)

def tensor_to_video(tensor):
    pass


def generate_bar_graphs_from_csv(data_dir)
    """
    Generate a set of bar graphs from a folder with csvs in it

    Args:
        data_dir (str): path to the directory with the data in it
    """
    csvs = [fp for fp in os.listdir(os.getcwd()) if fp.endswith(".csv")]

    for csv in csvs:
        df = pd.read_csv(csv)
        plt.figure()
        sns.barplot(data=df, x="function", y="time (s)", ci=None)

        plt.xticks(rotation=15)
        plt.savefig(f"{csv[:-4]}.png")
