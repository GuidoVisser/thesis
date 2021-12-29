from os import path
from typing import Union
import numpy as np
import torch
import cv2

from torch.nn.functional import grid_sample, l1_loss

from utils.utils import create_dirs
from InputProcessing.frameIterator import FrameIterator
from InputProcessing.maskHandler import MaskHandler

from models.third_party.RAFT import RAFT
from models.third_party.RAFT.utils.frame_utils import writeFlow, readFlow
from models.third_party.RAFT.utils.flow_viz import flow_to_image
from models.third_party.RAFT.utils.utils import InputPadder

class FlowHandler(object):
    def __init__(self, 
                 frame_iterator: FrameIterator, 
                 mask_iterator: MaskHandler,
                 output_dir: str,
                 raft_weights: str,
                 iters: int = 12,
                 normalize: bool = False,
                 forward_backward_threshold: float = 20.,
                 photometric_threshold: float = 20.,
                 device = "cuda") -> None:
        super().__init__()
        
        self.device = device
        self.iters  = iters
        self.forward_backward_threshold = forward_backward_threshold
        self.photometric_threshold      = photometric_threshold
        self.raft = self.initialize_raft(raft_weights)
        self.do_normalize = normalize

        self.output_dir= output_dir
        create_dirs(path.join(self.output_dir, "forward", "flow"), 
                    path.join(self.output_dir, "forward", "png"),
                    path.join(self.output_dir, "backward", "flow"),
                    path.join(self.output_dir, "backward", "png"),
                    path.join(self.output_dir, "confidence"),
                    path.join(self.output_dir, "dynamics_mask"))

        self.frame_iterator     = frame_iterator
        self.mask_iterator      = mask_iterator
        self.padder             = InputPadder(self.frame_iterator.frame_size)

        if not path.exists(path.join(self.output_dir, f"forward/flow/00000.flo")):
            self.max_value = 0.
            self.calculate_full_video_flow()
            with open(f"{self.output_dir}/max_value.txt", "w") as f:
                f.write(str(self.max_value))
        else:
            with open(f"{self.output_dir}/max_value.txt", "r") as f:
                self.max_value = float(f.read())

        if not self.do_normalize:
            self.max_value = 1.

    @torch.no_grad()
    def __getitem__(self, frame_idx: Union[int, slice]):
        """
        Calculate the forward flow between the frame at `frame_idx` and the next after being stabalized using 
        a perspective warp with the homography
        """

        # Get object masks
        _, object_masks = self.mask_iterator[frame_idx]
        object_masks = object_masks[:, 0]

        N_objects = object_masks.shape[0]

        if isinstance(frame_idx, slice):
            flows, confs, dynamics_masks = [], [], []
            for idx in range(frame_idx.start or 0, frame_idx.stop or len(self), frame_idx.step or 1):
                flows.append(torch.from_numpy(readFlow(path.join(self.output_dir, f"forward/flow/{idx:05}.flo"))).permute(2, 0, 1))
                confs.append(torch.from_numpy(cv2.imread(path.join(self.output_dir, f"confidence/{idx:05}.png"), cv2.IMREAD_GRAYSCALE)) / 255.)
                dynamics_masks.append(torch.from_numpy(cv2.imread(path.join(self.output_dir, f"dynamics_mask/{idx:05}.png"), cv2.IMREAD_GRAYSCALE)) / 255.)
            flow = torch.stack(flows, dim=-3)
            conf = torch.stack(confs, dim=-3)
            dynamics_mask = torch.stack(dynamics_masks, dim=-3)
        else:
            frame_path = path.join(self.output_dir, f"forward/flow/{frame_idx:05}.flo")
            flow = torch.from_numpy(readFlow(frame_path)).permute(2, 0, 1)
            conf = torch.from_numpy(cv2.imread(path.join(self.output_dir, f"confidence/{frame_idx:05}.png"), cv2.IMREAD_GRAYSCALE)) / 255.
            dynamics_mask = torch.from_numpy(cv2.imread(path.join(self.output_dir, f"dynamics_mask/{frame_idx:05}.png"), cv2.IMREAD_GRAYSCALE)) / 255.

        # normalize 
        if self.do_normalize:
            flow /= self.max_value

        # background_mask = 1 - torch.minimum(torch.sum(object_masks, dim=0), torch.ones(object_masks.shape[1:]))
        # dynamics_mask = background_mask * dynamics_mask

        conf = torch.stack([conf]*N_objects, dim=0) * object_masks

        # get flow of objects and background
        object_flow = self.get_object_flow(flow, object_masks)

        return flow, conf, object_flow, dynamics_mask
        
    @torch.no_grad()
    def calculate_full_video_flow(self):

        for frame_idx in range(len(self.frame_iterator) - 1):
            print(f"Calculating Optical Flow: {frame_idx} / {len(self.frame_iterator) - 1}")

            # Define images
            image0 = self.get_image(frame_idx)
            image1 = self.get_image(frame_idx + 1)

            h, w, _ = image0.shape

            # if self.aligned:
            #     # Align the frames using the pre-calculated homography
            #     image1, image0 = self.homography_handler.align_frames([image0, image1], indices=[frame_idx, frame_idx + 1])

            # Prepare images for use with RAFT
            image0 = self.prepare_image_for_raft(image0)
            image1 = self.prepare_image_for_raft(image1)

            padder = InputPadder(image0.shape)
            image0, image1 = padder.pad(image0, image1)

            # Calculate forward and backward optical flow
            _, forward_flow  = self.raft(image0, image1, iters=self.iters, test_mode=True)
            _, backward_flow = self.raft(image1, image0, iters=self.iters, test_mode=True)

            # Get the flow confidence
            conf = self.get_confidence(image0, image1, forward_flow, backward_flow)

            del backward_flow

            # remove batch dimension from RAFT
            forward_flow = forward_flow[0]

            # Recover original perspective
            forward_flow = padder.unpad(forward_flow)
            conf         = padder.unpad(conf)

            # Construct dynamics mask for the scene
            flow_magnitude = torch.sqrt(torch.square(forward_flow[0]) + torch.square(forward_flow[1]))
            flow_magnitude = flow_magnitude / torch.max(flow_magnitude)

            dynamics_mask = torch.where(flow_magnitude > torch.mean(flow_magnitude), torch.ones_like(flow_magnitude), torch.zeros_like(flow_magnitude))

            # if self.aligned:
            #     t = [-self.homography_handler.xmin, 
            #          -self.homography_handler.ymin]

            #     forward_flow = forward_flow[:, t[1]:h+t[1], t[0]:w+t[0]]
            #     conf         = conf[t[1]:h+t[1], t[0]:w+t[0]]

            self.max_value = max(self.max_value, torch.max(torch.abs(forward_flow)).item())

            forward_flow = forward_flow.permute(1, 2, 0).cpu().numpy()
            conf         = conf.cpu().numpy()
            dynamics_mask = torch.stack([dynamics_mask * 255]*3, dim=2).byte().cpu().numpy()
            writeFlow(path.join(self.output_dir, f"forward/flow/{frame_idx:05}.flo"), forward_flow)
            cv2.imwrite(path.join(self.output_dir, f"forward/png/{frame_idx:05}.png"), flow_to_image(forward_flow, convert_to_bgr=True))
            cv2.imwrite(path.join(self.output_dir, f"confidence/{frame_idx:05}.png"), np.expand_dims(conf, 2) * 255)
            cv2.imwrite(path.join(self.output_dir, f"dynamics_mask/{frame_idx:05}.png"), dynamics_mask)
            
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
        photometric_conf      = self.calculate_photometric_confidence(image0, image1, forward)

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
        ones  = torch.ones_like(photometric_error)
        zeros = torch.zeros_like(photometric_error)
        mask  = torch.where(photometric_error < self.photometric_threshold, ones, zeros)

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
        ones  = torch.ones_like(error)
        zeros = torch.zeros_like(error)
        conf  = torch.maximum(ones - error / self.forward_backward_threshold, zeros)

        # delete unnecessary tensors to save memory
        del ones, zeros

        return conf

    def get_object_flow(self, flow, masks):
        """
        Return the object flow of all objects in the scene
        """
        N_objects =  masks.shape[0]

        object_flow = torch.stack([flow]*N_objects)
        object_flow  = object_flow * masks

        return object_flow

    @staticmethod
    def apply_flow(tensor: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp a tensor with shape [B, C, H, W] with an optical flow field with shape [B, 2, H, W]

        Args:
            tensor (torch.Tensor)
            flow   (torch.Tensor)

        Returns:
            tensor (torch.Tensor): the warped input tensor
        """
        batch_size, _, h, w = tensor.size()

        # Calculate a base grid that functions as an identity sampler
        horizontal = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(batch_size, 1, h, w)
        vertical   = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        base_grid  = torch.cat([horizontal, vertical], dim=1).to(tensor.device)

        # calculate a Delta grid based on the flow that offsets the base grid
        flow_grid = torch.cat([flow[:, 0:1, :, :] / (w - 1.) / 2., 
                               flow[:, 1:2, :, :] / (h - 1.) / 2.], 1)

        # construct full grid
        grid = (base_grid + flow_grid).permute(0, 2, 3, 1)

        return grid_sample(tensor, grid, align_corners=True)

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
        _, forward_flow  = self.raft(image0, image1, iters=self.iters, test_mode=True)
        _, backward_flow = self.raft(image1, image0, iters=self.iters, test_mode=True)

        # Get the flow confidence
        object_mask = self.mask_iterator[frame_idx]
        conf = self.get_confidence(image0, image1, forward_flow, backward_flow, object_mask)

        return forward_flow, conf

    def __len__(self):
        return len(self.frame_iterator) - 1

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

    def initialize_raft(self, weights, small:bool=False, mixed_precision:bool=False, alternate_corr:bool=False):
        config = RaftNameSpace(
            small=small, 
            mixed_precision=mixed_precision, 
            alternate_corr=alternate_corr
        )

        model = torch.nn.DataParallel(RAFT(config))
        model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
        model = model.module
        model.to(self.device)
        model.eval()

        return model


class RaftNameSpace(object):
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def _get_kwargs(self):
        return self.__dict__.keys()
