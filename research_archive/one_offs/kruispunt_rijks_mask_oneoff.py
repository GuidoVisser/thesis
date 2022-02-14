from audioop import reverse
from utils.utils import create_dirs
import cv2
import numpy as np
from os import path, listdir
from utils.utils import create_dirs

def get_mask(object, frame_idx):
    name = path.join(root, object, f"mask/{frame_idx:05}.png")
    return cv2.imread(name) / 128. * 255.

def occlude(source, occlusions):
    for occlusion in occlusions:
        source = np.maximum(source - occlusion, 0)
    return source

if __name__ == "__main__":
    root = "datasets/Videos/Annotations/kruispunt_rijks_all"

    T = len(listdir(path.join(root, "bus/mask")))

    dirs = [path.join(root, f"intrinsic/{layer:02}") for layer in range(4)]
    dirs.extend([path.join(root, f"extrinsic/{layer:02}") for layer in range(4)])
    create_dirs(*dirs)
        
    for t in range(T):
        bike = get_mask("fiets", t)[..., 2]
        bus = get_mask("bus", t)[..., 2]
        jaap_jelle = get_mask("high_quality", t)
        jaap = jaap_jelle[..., 1]
        jelle = jaap_jelle[..., 2]

        jelle_mask = np.ones_like(jelle)
        jelle_mask[:, jelle_mask.shape[1]//2:] -= 1
        jelle = jelle * jelle_mask

        # cv2.imshow("bike", bike)
        # cv2.imshow("bus", bus)
        # cv2.imshow("jaap", jaap)
        # cv2.imshow("jelle", jelle)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        layers = [bus, bike, jaap, jelle]
        for l in range(4):
            mask = layers[l]

            mask = occlude(mask, layers[l+1:])

            # cv2.imshow("the thing", mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            fp = path.join(root, f"extrinsic/{l:02}/{t:05}.png")
            cv2.imwrite(fp, mask)
            

        # cv2.imwrite(path.join(root, f"intrinsic/00/{t:05}.png"), bus)
        # cv2.imwrite(path.join(root, f"intrinsic/01/{t:05}.png"), bike)
        # cv2.imwrite(path.join(root, f"intrinsic/02/{t:05}.png"), jaap)
        # cv2.imwrite(path.join(root, f"intrinsic/03/{t:05}.png"), jelle)

