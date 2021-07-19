from InputProcessing.frameIterator import FrameIterator
from os import path, listdir
from PIL import Image
from types import SimpleNamespace
import numpy as np
import torch
import cv2
from models.RAFT.utils import frame_utils

from utils.utils import create_dirs
from models.RAFT import RAFT
from InputProcessing.FlowHandler.utils import RaftNameSpace
from InputProcessing.frameIterator import FrameIterator

from models.RAFT.utils.frame_utils import writeFlow
from models.RAFT.utils.flow_viz import flow_to_image
from models.RAFT.utils.utils import InputPadder

class FlowHandler(object):
    def __init__(self, 
                 frame_iterator: FrameIterator, 
                 output_dir: str,
                 device = "cuda") -> None:
        super().__init__()
        
        self.device = device
        self.raft = self.initialize_raft()

        self.output_dir= output_dir
        create_dirs([path.join(self.output_dir, dir) for dir in ["flow", "png"]])

        self.frame_iterator = frame_iterator
        self.padder = InputPadder(self.frame_iterator.frame_size)

    def calculate_flow_for_video(self):
        """
        Calculate the optical flow for the entire video
        """

        for i in range(len(self.frame_iterator) - 1):
            image0 = cv2.cvtColor(self.frame_iterator[i], cv2.COLOR_BGR2RGB)
            image0 = torch.from_numpy(image0).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

            image1 = cv2.cvtColor(self.frame_iterator[i+1], cv2.COLOR_BGR2RGB)
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

            _, flow = self.raft(image0, image1, iters=12, test_mode=True)

            flow = flow[0].permute(1, 2, 0).cpu().numpy()

            writeFlow(path.join(self.output_dir, "flow", f"{i:05}.flo"), flow)
            cv2.imwrite(path.join(self.output_dir, "png", f"{i:05}.png"), flow_to_image(flow, convert_to_bgr=True))

    def load_flow_image(self, frame_idx):
        img = cv2.imread(path.join(self.output_dir, "png", f"{frame_idx:05}.png"))
        return img

    def initialize_raft(self, small:bool=False, mixed_precision:bool=False, alternate_corr:bool=False):
        config = RaftNameSpace(
            small=small, 
            mixed_precision=mixed_precision, 
            alternate_corr=alternate_corr
        )

        model = torch.nn.DataParallel(RAFT(config))
        model.load_state_dict(torch.load("models/weights/raft-things.pth"))
        model = model.module
        model.to(self.device)
        model.eval()

        return model
