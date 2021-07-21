from os import path, listdir
import numpy as np
import torch
import cv2

from torch.nn.functional import grid_sample

from utils.utils import create_dirs
from InputProcessing.BackgroundAttentionVolume.align_frames import align_two_frames
from InputProcessing.FlowHandler.utils import RaftNameSpace
from InputProcessing.frameIterator import FrameIterator

from models.RAFT import RAFT
from models.RAFT.utils.frame_utils import writeFlow, readFlow
from models.RAFT.utils.flow_viz import flow_to_image
from models.RAFT.utils.utils import InputPadder

class FlowHandler(object):
    def __init__(self, 
                 frame_iterator: FrameIterator, 
                 homographies: list,
                 output_dir: str,
                 iters: int = 12,
                 device = "cuda") -> None:
        super().__init__()
        
        self.device = device
        self.iters = iters
        self.raft = self.initialize_raft()

        self.output_dir= output_dir
        create_dirs(path.join(self.output_dir, "flow"), path.join(self.output_dir, "png"))

        self.frame_iterator = frame_iterator
        self.homographies = homographies
        self.padder = InputPadder(self.frame_iterator.frame_size)

    def calculate_flow_for_video(self, stabalized=False):
        """
        Calculate the optical flow for the entire video
        """

        for i in range(len(self.frame_iterator) - 1):
            
            if stabalized:
                flow, _ = self.calculate_flow_between_stabilized_frames(i)
            else:
                flow, _ = self.calculate_flow_between_frames(i)

            writeFlow(path.join(self.output_dir, "flow", f"{i:05}.flo"), flow)
            cv2.imwrite(path.join(self.output_dir, "png", f"{i:05}.png"), flow_to_image(flow, convert_to_bgr=True))

    def calculate_flow_between_frames(self, frame_idx):
        """
        Calculate the forward flow between two subsequent frames
        """
        image0 = self.get_image(frame_idx)
        image0 = self.prepare_image_for_raft(image0)

        image1 = self.get_image(frame_idx + 1)
        image1 = self.prepare_image_for_raft(image1)

        _, forward_flow = self.raft(image0, image1, iters=self.iters, test_mode=True)
        _, backward_flow = self.raft(image1, image0, iters=self.iters, test_mode=True)

        conf = self.get_flow_confidence(forward_flow, backward_flow)

        conf = conf.permute(1, 2, 0).cpu().numpy()
        flow = forward_flow[0].permute(1, 2, 0).cpu().numpy()

        forward_flow = forward_flow[0].permute(1, 2, 0).cpu().numpy()
        backward_flow = backward_flow[0].permute(1, 2, 0).cpu().numpy()

        cv2.imshow("flow", flow_to_image(flow, convert_to_bgr=True))
        cv2.imshow("forward", flow_to_image(forward_flow, convert_to_bgr=True))
        cv2.imshow("backward", flow_to_image(backward_flow, convert_to_bgr=True))
        cv2.imshow("confidence", conf)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return flow, conf


    def calculate_flow_between_stabilized_frames(self, frame_idx):
        """
        Calculate the forward flow between the frame at `frame_idx` and the next after being stabalized using 
        a perspective warp with the homography
        """
        image0 = self.get_image(frame_idx)
        image1 = self.get_image(frame_idx + 1)

        h, w, _ = image0.shape

        image1, image0, _, translation = align_two_frames(image0, image1, self.homographies[frame_idx])

        image0 = self.prepare_image_for_raft(image0)
        image1 = self.prepare_image_for_raft(image1)

        padder = InputPadder(image0.shape)
        image0, image1 = padder.pad(image0, image1)

        _, forward_flow = self.raft(image0, image1, iters=self.iters, test_mode=True)
        _, backward_flow = self.raft(image1, image0, iters=self.iters, test_mode=True)

        conf = self.get_flow_confidence(forward_flow, backward_flow)

        flow = padder.unpad(forward_flow)
        conf = padder.unpad(conf)
        flow = flow[:, :, translation[1]:h+translation[1], translation[0]:w+translation[0]]
        conf = conf[:, translation[1]:h+translation[1], translation[0]:w+translation[0]]

        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        conf = conf.permute(1, 2, 0).cpu().numpy()

        forward_flow = forward_flow[0].permute(1, 2, 0).cpu().numpy()
        backward_flow = backward_flow[0].permute(1, 2, 0).cpu().numpy()

        cv2.imshow("flow", flow_to_image(flow, convert_to_bgr=True))
        cv2.imshow("forward", flow_to_image(forward_flow, convert_to_bgr=True))
        cv2.imshow("backward", flow_to_image(backward_flow, convert_to_bgr=True))
        cv2.imshow("confidence", conf)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return flow, conf
        
    def get_flow_confidence(self, forward, backward):
        """
        Calculate the forward backward flow error using the euclidian distance 
        and use this to produce a confidence measurement for the optical flow
        """

        error = self.get_forward_backward_error(forward, backward)

        ones = torch.ones(error.size(), device=self.device)
        zeros = torch.zeros(error.size(), device=self.device)

        conf = torch.maximum(10*ones - error, zeros)

        del ones, zeros

        return conf


    def get_forward_backward_error(self, forward, backward):
        
        backward = grid_sample(backward, forward.permute(0, 2, 3, 1) * -1.)
        diff = forward + backward
        
        error = torch.sqrt(torch.sum(torch.square(diff), dim=1))

        return error

    def get_image(self, frame_idx):
        return cv2.cvtColor(self.frame_iterator[frame_idx], cv2.COLOR_BGR2RGB)
    
    def prepare_image_for_raft(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

    def load_flow_image(self, frame_idx):
        return cv2.imread(path.join(self.output_dir, "png", f"{frame_idx:05}.png"))

    def load_flow(self, frame_idx):
        return readFlow(path.join(self.output_dir, "flow", f"{frame_idx:05}.flo"))

    def convert_flow_to_image(self, flow, bgr=False):
        return flow_to_image(flow, convert_to_bgr=bgr)

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
