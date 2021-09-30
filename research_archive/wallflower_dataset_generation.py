import cv2
from os import path, listdir
from utils.utils import create_dir
import numpy as np

def select_frames():
    start = 220
    end   = 280

    video = "WavingTrees"
    file_root = f"wallflower/{video}"
    file_dest = f"datasets/wallflower/{video}"

    create_dir(file_dest)

    for i in range(start, end):
        fn = f"b{i:05}.bmp"
        img = cv2.imread(path.join(file_root, fn))
        cv2.imwrite(path.join(file_dest, f"{(i-start):05}.png"), img)

def mivos2annotation(img_path: str, num_masks: int) -> None:

    img = cv2.imread(img_path)

    dummie = np.where((np.sum(img, axis=2) > 128), 0, 1).astype(np.uint8)
    dummie_inv = np.expand_dims(1 - dummie, 2)

    img *= np.stack([dummie]*3, 2)
    img[:, :, 0:1] = dummie_inv

    for i in range(num_masks):

        mask = np.where((img[:, :, 2-i] > 0), 255, 0)

        fp = "/".join(img_path.split('/')[:-1])
        fn = img_path.split('/')[-1]
        create_dir(path.join(fp, f"{i:02}"))
        cv2.imwrite(path.join(fp, f"{i:02}", fn), mask)


if __name__ == "__main__":
    video = "Bootstrap"
    file_root = f"datasets/wallflower/{video}"
    
    img_path = path.join(file_root, "Annotations", "00000.png")
    mivos2annotation(img_path, 3)