from models.third_party.RAFT.utils.flow_viz import flow_to_image
from models.third_party.RAFT.utils.frame_utils import readFlow
from utils.utils import create_dir
import cv2
import numpy as np
from os import path, listdir

class Compositer(object):
    def __init__(self, root: str) -> None:
        super().__init__()

        self.root = root
        self.N_layers = len(listdir(root))
        self.N_frames = len(listdir(path.join(root, "layers/00")))

        self.save_dir = path.join(self.root, "composites")
        create_dir(self.save_dir)

    def _get_layer_img(self, frame_idx: int, layer_idx: int) -> np.array:
        """
        Get the RGBA image of a layer and frame

        Args:
            frame_idx (int): index of frame in video
            layer_idx (int): index of the object layer
        
        Returns:
            rgba (np.array(H, W, 4)): rgba values of the frame and layer
        """
        return cv2.imread(path.join(self.root, "layers", f"{layer_idx:02}", f"{frame_idx:05}.png"), cv2.IMREAD_UNCHANGED) / 255.

    def _get_layer_flow(self, frame_idx: int, layer_idx: int) -> np.array:
        """
        Get the optical flow values of a layer and frame

        Args:
            frame_idx (int): index of frame in video
            layer_idx (int): index of the object layer

        Returns:
            flow (np.array [H, W, 2]): optical flow values of the layer and frame
        """
        return readFlow(path.join(self.root, "flow/flo", f"{layer_idx:02}", f"{frame_idx:05}.flo"))

    def composite_frame(self, frame_idx: int, layers: list) -> np.array:

        composite = self._get_layer_img(frame_idx, layers[0])
        alpha_composite = composite[..., 3:]
        for layer in layers[1:]:
            
            new_layer = self._get_layer_img(frame_idx, layer)
            alpha = new_layer[..., 3:]

            composite = (1 - alpha) * composite + alpha * new_layer
            alpha_composite = (1 - alpha) * alpha_composite + alpha

        composite[..., 3:] = alpha_composite

        return composite

    def composite_flow_frame(self, frame_idx: int, layers: list) -> np.array:

        composite = self._get_layer_flow(frame_idx, layers[0])

        for layer in layers[1:]:
            
            new_layer = self._get_layer_flow(frame_idx, layer)
            alpha = self._get_layer_img(frame_idx, layer)[..., 3:]

            composite = (1 - alpha) * composite + alpha * new_layer

        flow_img = flow_to_image(composite, convert_to_bgr=True)
        
        return flow_img

    def composite_and_save_frames(self, frames: list, layers: list, type="images") -> None:

        assert type in ["images", "flow"]

        dir_name = type + "_" + "_".join([f"{layer:02}" for layer in layers])
        create_dir(path.join(self.save_dir, dir_name))

        for frame in frames:

            if type == "images":
                img = self.composite_frame(frame, layers) * 255
            else:
                img = self.composite_flow_frame(frame, layers)

            cv2.imwrite(path.join(self.save_dir, dir_name, f"{frame:05}.png"), img)

if __name__ == "__main__":

    video = "kruispunt_rijks"
    root  = f"results/layer_decomposition_dynamic/{video}/decomposition/final"
    compositer = Compositer(root)

    # frames = sorted(listdir(path.join(root, "layers/00")))
    frames = list(range(50))
    layers = [0, 2, 3]

    compositer.composite_and_save_frames(frames, layers, "flow")