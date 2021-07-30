from os import path, listdir
import numpy as np
from numpy.core.shape_base import stack
import torch
import cv2

from torch.nn.functional import grid_sample, l1_loss
from InputProcessing import frameIterator

from utils.utils import create_dirs
from InputProcessing.BackgroundAttentionVolume.align_frames import align_two_frames
from InputProcessing.FlowHandler.utils import RaftNameSpace
from InputProcessing.frameIterator import FrameIterator
from InputProcessing.MaskPropagation import maskHandler

from models.third_party.RAFT import RAFT, update
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
        
        create_dirs(path.join(self.output_dir, "forward", "flow"), 
                    path.join(self.output_dir, "forward", "png"),
                    path.join(self.output_dir, "backward", "flow"),
                    path.join(self.output_dir, "backward", "png"),
                    path.join(self.output_dir, "confidence"))

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
        # Define images
        image0 = self.get_image(frame_idx)
        image1 = self.get_image(frame_idx + 1)
        
        # Prepare images for use with RAFT
        image0 = self.prepare_image_for_raft(image0)
        image1 = self.prepare_image_for_raft(image1)

        # Calculate forward and backward optical flow
        _, forward_flow = self.raft(image0, image1, iters=self.iters, test_mode=True)
        _, backward_flow = self.raft(image1, image0, iters=self.iters, test_mode=True)

        # Get the flow confidence
        object_mask = self.mask_iterator[frame_idx]
        conf = self.get_confidence(image0, image1, forward_flow, backward_flow, object_mask)

        return forward_flow, conf

    @torch.no_grad()
    def __getitem__(self, frame_idx):
        """
        Calculate the forward flow between the frame at `frame_idx` and the next after being stabalized using 
        a perspective warp with the homography
        """

        mask = self.mask_iterator.get_binary_masks(frame_idx)
        object_masks = torch.from_numpy(mask).to(self.device)

        N_objects = object_masks.shape[0]

        frame_path = path.join(self.output_dir, f"forward/flow/{frame_idx:05}.flo")
        if path.exists(frame_path):
            forward_flow = torch.from_numpy(readFlow(frame_path)).permute(2, 0, 1).to(self.device)
            conf = torch.from_numpy(cv2.imread(path.join(self.output_dir, f"confidence/{frame_idx:05}.png"), cv2.IMREAD_GRAYSCALE)).to(self.device)
        else:

            # Define images
            image0 = self.get_image(frame_idx)
            image1 = self.get_image(frame_idx + 1)

            h, w, _ = image0.shape

            # Align the frames using the pre-calculated homography
            image1, image0, _, translation = align_two_frames(image0, image1, self.homographies[frame_idx])

            # Prepare images for use with RAFT
            image0 = self.prepare_image_for_raft(image0)
            image1 = self.prepare_image_for_raft(image1)

            padder = InputPadder(image0.shape)
            image0, image1 = padder.pad(image0, image1)

            # Calculate forward and backward optical flow
            _, forward_flow = self.raft(image0, image1, iters=self.iters, test_mode=True)
            _, backward_flow = self.raft(image1, image0, iters=self.iters, test_mode=True)

            # Get the flow confidence
            conf = self.get_confidence(image0, image1, forward_flow, backward_flow)

            del backward_flow

            # remove batch dimension from RAFT
            forward_flow = forward_flow[0]

            # Recover original perspective
            forward_flow = padder.unpad(forward_flow)
            conf = padder.unpad(conf)
            forward_flow = forward_flow[:, translation[1]:h+translation[1], translation[0]:w+translation[0]]
            conf = conf[translation[1]:h+translation[1], translation[0]:w+translation[0]]

            # save results
            forward_flow_copy = torch.clone(forward_flow).permute(1, 2, 0).cpu().numpy()
            conf_copy = torch.clone(conf).cpu().numpy()

            writeFlow(path.join(self.output_dir, f"forward/flow/{frame_idx:05}.flo"), forward_flow_copy)
            cv2.imwrite(path.join(self.output_dir, f"forward/png/{frame_idx:05}.png"), flow_to_image(forward_flow_copy))
            cv2.imwrite(path.join(self.output_dir, f"confidence/{frame_idx:05}.png"), np.expand_dims(conf_copy, 2) * 255)
            
            del forward_flow_copy

        conf = torch.stack([conf]*N_objects, dim=0) * object_masks

        # get flow of objects and background
        object_flow, background_flow = self.get_object_and_background_flow(forward_flow, object_masks)

        return forward_flow, conf, object_flow, background_flow
        
    def get_confidence(self, image0, image1, forward, backward):
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
            image0      (torch.Tensor[C, H, W])
            image1      (torch.Tensor[C, H, W])
            forward     (torch.Tensor[2, H, W])
            backward    (torch.Tensor[2, H, W])
            object_mask (torch.Tensor[N, H, W])

        Returns:
            weights     (torch.Tensor[N, H, W])
        """

        # Calculate the different confidence terms
        forward_backward_conf = self.calculate_forward_backward_confidence(forward, backward)
        photometric_conf = self.calculate_photometric_confidence(image0, image1, forward)

        # N_objects, _, _ = object_masks.size()

        # # Stack confidence terms in layer dimension
        # forward_backward_conf = torch.stack([forward_backward_conf]*N_objects, dim=0)
        # photometric_conf = torch.stack([photometric_conf]*N_objects, dim=0)

        # return pixelwise weights for flow reconstruction confidence
        return forward_backward_conf * photometric_conf #* object_masks

    def calculate_photometric_confidence(self, image0, image1, flow):
        """
        Calculate the photometric error between `image0` and `image1` that is warped back into image0
        using the (forward) optical flow between the two frames.

        Returns a mask that is 1 where the photometric error is lower than a threshold (beta) and 0 everywhere else.

        Args:
            image0 (torch.Tensor[C, H, W])
            image1 (torch.Tensor[C, H, W])
            flow   (torch.Tensor[2, H, W])

        Returns:
            mask (torch.Tensor[H, W])
        """

        # warp second image in pair using the flow
        image1 = self.apply_flow(image1, flow)

        # Calculate the photometric error between I_t and warped I_(t+1)
        photometric_error = l1_loss(image0, image1, reduction="none")
        photometric_error = torch.sum(photometric_error, dim=1)

        # remove batch dimension from RAFT
        photometric_error = photometric_error[0]

        # Construct a mask based on the photometric error and threshold
        ones = torch.ones_like(photometric_error)
        zeros = torch.zeros_like(photometric_error)
        mask =  torch.where(photometric_error < self.photometric_threshold, ones, zeros)

        # delete unnecessary tensors to save memory
        del ones, zeros

        return mask

    def calculate_forward_backward_confidence(self, forward, backward):
        """
        Calculate the forward backward error of the optical flow 

        This error term is defined as the distance between point x and x' for every pixel, where x' is
        calculated by applying the forward flow on x and then the backward flow on the result

        x'(p) = backward( forward( x(p) ) )
        
        Args:
            forward  (torch.Tensor[2, H, W]): forward optical flow field
            backward (torch.Tensor[2, H, W]): backward optical flow field

        Returns:
            error (torch.Tensor[H, W]): pixelwise forward-backward error
        """

        # warp backward flow such that it aligns with forward flow
        backward = self.apply_flow(backward, forward)  

        # calculate magnitude of resulting difference vector at every pixel
        error = torch.sqrt(torch.sum(torch.square(forward + backward), dim=1))

        # remove RAFT batch dimension
        error = error[0]
        
        # define the confidence tensor
        ones = torch.ones_like(error)
        zeros = torch.zeros_like(error)
        conf = torch.maximum(ones - error / self.forward_backward_threshold, zeros)

        # delete unnecessary tensors to save memory
        del ones, zeros

        return conf

    def get_object_and_background_flow(self, flow, masks):
        """
        Return the object flow of all objects in the scene
        """
        N_objects, _, _ =  masks.shape

        stacked_flow = torch.stack([flow]*N_objects)
        object_flow = stacked_flow * masks

        background_mask = 1. - torch.sum(masks, dim=0)
        background_mask = torch.maximum(background_mask, torch.zeros_like(background_mask))
        background_flow = flow * background_mask

        return object_flow, background_flow


    @staticmethod
    def apply_flow(tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp a tensor with shape [B, C, H, W] with an optical flow field with shape [B, 2, H, W]

        Args:
            tensor (torch.Tensor)
            flow (torch.Tensor)

        Returns:
            tensor (torch.Tensor): the warped input tensor
        """
        _, _, h, w = tensor.size()

        # Calculate a base grid that functions as an identity sampler
        horizontal = torch.linspace(-1.0, 1.0, flow.size(3)).view(1, 1, 1, flow.size(3)).expand(flow.size(0), 1, flow.size(2), flow.size(3))
        vertical   = torch.linspace(-1.0, 1.0, flow.size(2)).view(1, 1, flow.size(2), 1).expand(flow.size(0), 1, flow.size(2), flow.size(3))
        base_grid = torch.cat([horizontal, vertical], dim=1).to(tensor.device)

        # calculate a Delta grid based on the flow that offsets the base grid
        flow_grid = torch.cat([flow[:, 0:1, :, :] / (w - 1.) / 2., 
                               flow[:, 1:2, :, :] / (h - 1.) / 2.], 1)

        # construct full grid
        grid = (base_grid + flow_grid).permute(0, 2, 3, 1)

        return grid_sample(tensor, grid, align_corners=True)

    def __len__(self):
        return len(self.frame_iterator)

    def get_image(self, frame_idx):
        return cv2.cvtColor(self.frame_iterator.get_np_frame(frame_idx), cv2.COLOR_BGR2RGB)
    
    def prepare_image_for_raft(self, img):
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

    def prepare_flow_for_cv2(self, flow):
        return flow[0].permute(1, 2, 0).cpu().numpy()

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
        model.load_state_dict(torch.load("models/third_party/weights/raft-things.pth"))
        model = model.module
        model.to(self.device)
        model.eval()

        return model
