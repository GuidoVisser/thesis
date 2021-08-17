import cv2
from os import path, listdir
import glob
import numpy as np
import imageio

def create_decomposite_demo(root):

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
    video_path = path.join(root, "demo.gif")
    imageio.mimsave(video_path, img_array, format="GIF", fps=25)

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