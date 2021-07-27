from os import path, listdir
import numpy as np
import torch
import cv2

from torch.nn.functional import grid_sample, l1_loss

from utils.utils import create_dirs
from InputProcessing.BackgroundAttentionVolume.align_frames import align_two_frames
from InputProcessing.FlowHandler.utils import RaftNameSpace
from InputProcessing.frameIterator import FrameIterator
from InputProcessing.MaskPropagation import maskHandler

from models.third_party.RAFT import RAFT
from models.third_party.RAFT.utils.frame_utils import writeFlow, readFlow
from models.third_party.RAFT.utils.flow_viz import flow_to_image
from models.third_party.RAFT.utils.utils import InputPadder

class FlowHandler(object):
    def __init__(self, 
                 frame_iterator: FrameIterator, 
                 mask_iterator: maskHandler,
                 homographies: list,
                 output_dir: str,
                 iters: int = 12,
                 forward_backward_threshold: float = 20.,
                 photometric_threshold: float = 20.,
                 device = "cuda") -> None:
        super().__init__()
        
        self.device = device
        self.iters = iters
        self.forward_backward_threshold = forward_backward_threshold
        self.photometric_threshold = photometric_threshold
        self.raft = self.initialize_raft()

        self.output_dir= output_dir
        create_dirs(path.join(self.output_dir, "flow"), path.join(self.output_dir, "png"))

        self.frame_iterator = frame_iterator
        self.mask_iterator = mask_iterator
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

        return flow, conf
        
    def get_reconstruction_weights(self, image0, image1, forward, backward, object_masks):
        """
        Calculate an alhpa mask for every object that serves as a weight for the flow reconstruction
        The alpha mask at timestep t is defined as follows:

        W_lr_t * W_p_t * M_t

        where:
            W_lr_t is a confidence term based on the forward-backward error of the optical flow estimation
                W_lr_t  = max( 1 - e_lr / threshold, 0 )
            
            W_p_t is a confidence term based on the photometric error between I_t and and I_(t+1) warped by the optical flow
                W_p_t   =   1   if ||Warp( I_(t+1); F_(t,t+1) ) - I_t|| < threshold
                            0   otherwise
            
            M_t is the object mask at timestep t    

        Args:
            image0      (torch.Tensor[B, C, H, W])
            image1      (torch.Tensor[B, C, H, W])
            forward     (torch.Tensor[B, 2, H, W])
            backward    (torch.Tensor[B, 2, H, W])
            object_mask (torch.Tensor[B, N, 1, H, W])

        Returns:
            weights     (torch.Tensor[B, N, 1, H, W])
        """

        forward_backward_conf = self.get_flow_confidence(forward, backward)
        photometric_conf = self.calculate_photometric_error(image0, image1)

        _, N_objects, _, _, _ = object_masks.size()

        forward_backward_conf = torch.stack([forward_backward_conf]*N_objects, dim=1)
        photometric_conf = torch.stack([photometric_conf]*N_objects, dim=1)

        return forward_backward_conf * photometric_conf * object_masks  


    def get_flow_confidence(self, forward, backward):
        """
        Calculate the forward backward flow error using the euclidian distance 
        and use this to produce a confidence measurement for the optical flow
        """

        error = self.get_forward_backward_error(forward, backward)

        ones = torch.ones(error.size(), device=self.device)
        zeros = torch.zeros(error.size(), device=self.device)

        conf = torch.maximum(ones - error / self.forward_backward_threshold, zeros)

        del ones, zeros

        return conf


    def calculate_photometric_error(self, image0, image1, flow):
        """
        Calculate the photometric error between `image0` and `image1` that is warped back into image0
        using the (forward) optical flow between the two frames.

        Returns a mask that is 1 where the photometric error is lower than a threshold (beta) and 0 everywhere else.

        Args:
            image0 (torch.Tensor[B, C, H, W])
            image1 (torch.Tensor[B, C, H, W])
            flow   (torch.Tensor[B, 2, H, W])

        Returns:
            mask (torch.Tensor[B, H, W])
        """
        image1 = self.apply_flow(image1, flow)

        photometric_error = l1_loss(image0, image1)

        ones = torch.ones(photometric_error.size(), device=self.device)
        zeros = torch.zeros(photometric_error.size(), device=self.device)

        mask =  torch.where(photometric_error < self.photometric_threshold, ones, zeros)

        del ones, zeros

        return mask


    def get_forward_backward_error(self, forward, backward):
        """
        Calculate the forward backward error of the optical flow 

        This error term is defined as the distance between point x and x' for every pixel, where x' is
        calculated by applying the forward flow on x and then the backward flow on the result

        x'(p) = backward( forward( x(p) ) )
        
        Args:
            forward  (torch.Tensor[B, 2, H, W]): forward optical flow field
            backward (torch.Tensor[B, 2, H, W]): backward optical flow field

        Returns:
            error (torch.Tensor[B, H, W]): pixelwise forward-backward error
        """

        # warp backward flow such that it aligns with forward flow
        backward = self.apply_flow(backward, forward)  

        # calculate magnitude of resulting difference vector at every pixel
        error = torch.sqrt(torch.sum(torch.square(forward + backward), dim=1))

        return error

    def apply_flow(self, tensor, flow):
        """
        Warp a tensor with shape [B, C, H, W] with an optical flow field with shape [B, 2, H, W]

        Args:
            tensor (torch.Tensor)
            flow (torch.Tensor)

        Returns:
            tensor (torch.Tensor): the warped input tensor
        """
        _, _, h, w = tensor.size()

        horizontal = torch.linspace(-1.0, 1.0, flow.size(3)).view(1, 1, 1, flow.size(3)).expand(flow.size(0), 1, flow.size(2), flow.size(3))
        vertical   = torch.linspace(-1.0, 1.0, flow.size(2)).view(1, 1, flow.size(2), 1).expand(flow.size(0), 1, flow.size(2), flow.size(3))
        base_grid = torch.cat([horizontal, vertical], dim=1).to(self.device)

        flow_grid = torch.cat([flow[:, 0:1, :, :] / (w - 1.) / 2., 
                               flow[:, 1:2, :, :] / (h - 1.) / 2.], 1)

        grid = (base_grid + flow_grid).permute(0, 2, 3, 1)

        return grid_sample(tensor, grid)


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
