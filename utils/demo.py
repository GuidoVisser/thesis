import cv2
from os import path, listdir
import glob
import numpy as np
import imageio
from typing import Union

def create_decomposite_demo(roots: Union[str, list], fps=10):

    if isinstance(roots, str):
        video_path = path.join(roots, "demo.gif")
        roots = [roots]
    else:
        video_path = path.join(roots[0], "demo.gif")

    combined_array = []
    for root in roots:
        img_array = []
        for fn in sorted(glob.glob(path.join(root, "ground_truth/*.png"))):
            img = cv2.imread(fn)
            img_array.append(img)

        for i, fn in enumerate(sorted(glob.glob(path.join(root, "background/*.png")))):
            img = cv2.imread(fn)
            img_array[i] = np.concatenate((img_array[i], img), axis=1)

        for object in sorted(listdir(path.join(root, "foreground"))):
            for i, fn in enumerate(sorted(glob.glob(path.join(root, "foreground", object, "*.png")))):
                img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
                alpha = np.stack([img[:, :, 3]]*3, axis=2) / 255
                img = np.uint8(img[:, :, :3] * alpha)
                img_array[i] = np.concatenate((img_array[i], img), axis=1)
        

        img_array = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in img_array]
        img_array = np.stack(img_array)

        combined_array.append(img_array)
    
    combined_array = np.concatenate(combined_array, axis=1)
    imageio.mimsave(video_path, combined_array, format="GIF", fps=fps)

    # h, w, c = img_array[0].shape 
    # out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"X264"), 15, (w, h))

    # for img in img_array:
    #     out.write(img)
    # out.release()

    # print("Saving GIF file")
    # with imageio.get_writer(video_path, mode="I") as writer:
    #     for idx, frame in enumerate(img_array):
    #         print("Adding frame to GIF file: ", idx + 1)
    #         writer.append_data(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    fps = 5

    file_paths = [
        "results/layer_decomposition/scooter-black_10_20_18_24_28/decomposition/inference",
        "results/layer_decomposition_dynamic/scooter-black_10_28_00_02_20/decomposition/inference",
        "results/layer_decomposition_dynamic/scooter-black_10_28_00_02_54/decomposition/inference",
        "results/layer_decomposition_dynamic/scooter-black_10_28_00_06_50/decomposition/inference",
        "results/layer_decomposition_dynamic/scooter-black_10_28_01_56_16/decomposition/inference",
        "results/layer_decomposition_dynamic/scooter-black_10_28_02_01_09/decomposition/inference",
        "results/layer_decomposition_dynamic/scooter-black_10_28_03_43_34/decomposition/inference",
        "results/layer_decomposition_dynamic/scooter-black_10_28_04_03_45/decomposition/inference",
    ]    

    create_decomposite_demo(file_paths, fps)
