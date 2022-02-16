import cv2
import numpy as np
from os import path, listdir
from compositing import Compositer
import torch

def get_masks(layer, frame):
    object_mask = cv2.imread(f"datasets/Experimental/Dynamics/nescio_2/nescio_2_8752504/decomposition/final/alpha/{layer:02}/{frame:05}.png")
    object_mask = torch.from_numpy(object_mask) / 255.

    if layer == 0:
        dynamics_mask = torch.where(object_mask > 0.1, torch.zeros_like(object_mask), torch.ones_like(object_mask)) 
    else:
        dynamics_mask = torch.where(object_mask > 0.1, torch.ones_like(object_mask), torch.zeros_like(object_mask)) 
    dynamics_mask = dynamics_mask.numpy()
    object_mask = object_mask.numpy()

    return object_mask, dynamics_mask

def get_binary_object_mask(layer, frame):
    object_mask = cv2.imread(f"datasets/Videos/Annotations/nescio_2/{layer:02}/{frame:05}.png") / 255.
    return object_mask.astype('uint8')

def alpha_based_dyn_masks():
    root = "datasets/Experimental/Dynamics/nescio_2/nescio_2_8752504/decomposition/final"
    compositer = Compositer(root)

    # compositer.composite_and_save_frames(list(range(compositer.N_frames)), [0, 1, 2, 3])
    for frame in range(compositer.N_frames):
        object_masks = []
        dynamics_masks = []
        for layer in range(2):
            object_mask, dynamics_mask = get_masks(layer, frame)
            object_masks.append(object_mask)
            dynamics_masks.append(dynamics_mask)

        img = cv2.imread(f"datasets/Experimental/Images/nescio_2/{frame:05}.png")
        for i in range(len(dynamics_masks)):
            dynamics_masks[i] = cv2.resize(dynamics_masks[i], (img.shape[1], img.shape[0]))
            dynamics_masks[i] = dynamics_masks[i].astype('uint8')

        dynamics_mask = np.minimum(np.sum(np.stack(dynamics_masks), axis=0), 1).astype('uint8')

        for i in range(2):
            bin_mask = get_binary_object_mask(i, frame)
            dynamics_mask = dynamics_mask * (1 - bin_mask)
            dynamics_mask = np.maximum(dynamics_mask, 0)
        
        dynamics_mask = (1 - dynamics_mask) * 255

        dynamics_mask = cv2.erode(dynamics_mask, kernel=np.ones((10, 10)), iterations=1)

        cv2.imshow("d", dynamics_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite(f"datasets/Experimental/Dynamics/nescio_2/{frame:05}.png", dynamics_mask)

if __name__ == "__main__":
    
    alpha_based_dyn_masks()
    # for frame_idx in range(60):
    #     root = "datasets/Experimental/Dynamics/nescio_2"
    #     mask_root = "datasets/Videos/Annotations/nescio_2"

    #     object_masks = []
    #     for layer in sorted(listdir(mask_root)):
    #         object_masks.append(cv2.imread(path.join(mask_root, layer, f"{frame_idx:05}.png")) / 255.) 

    #     object_mask = np.minimum(np.sum(np.stack(object_masks), axis=0), np.ones_like(object_masks[0]))

    #     dynamics_mask = cv2.imread(path.join(root, f"{frame_idx:05}.png"))

    #     dynamics_mask = np.minimum((255 - dynamics_mask) + (object_mask * 255).astype('uint8'), 255)

    #     dynamics_mask = cv2.dilate(dynamics_mask, kernel=np.ones((5,5)))

    #     # cv2.imshow("d",dynamics_mask)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     cv2.imwrite(path.join(root, f"{frame_idx:05}.png"), dynamics_mask)