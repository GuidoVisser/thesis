import os
import numpy as np
import torch

from mask_propagation import propagate_mask_through_video
from utils.flow_utils import normalize_optical_flow
from utils.video_utils import create_masked_video
from models.RAFT.utils.utils import InputPadder


def calculate_batch_flow(flow_model, source_batch, target_batch):

    _, flow = flow_model(source_batch, target_batch, iters=1, test_mode=True)

    return flow


@torch.no_grad()
def test_vae(model, extra_models, data_loader):
    """
    Function for testing a model on a dataset.
    Inputs:
        model - VAE model to test
        data_loader - Data Loader for the dataset you want to test on.
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """
    flow_model = extra_models[0]

    # initialize padder object for RAFT
    sample_frame, _, _ = next(iter(data_loader))
    raft_padder = InputPadder(sample_frame.size())
    del sample_frame

    average_bpd = 0
    average_rec_loss = 0
    average_reg_loss = 0

    for step, batch in enumerate(data_loader):
        
       # getting batch
        frames, masks, flow = [input.to(model.device) for input in batch]

        current_mask, masks = torch.split(masks, [1, masks.size()[2]-1], dim=2)
        current_mask = raft_padder.pad(current_mask.squeeze(2))[0]

        count = 1
        while frames.size()[2] > 1:
            
            current_frame, frames = torch.split(frames, [1, frames.size()[2]-1], dim=2)
            next_mask, masks = torch.split(masks, [1, masks.size()[2]-1], dim=2)

            current_frame = raft_padder.pad(current_frame.squeeze(2))[0]
            next_mask = raft_padder.pad(next_mask.squeeze(2))[0]

            # prepare optical flow
            if flow is not None:
                current_flow, flow = torch.split(flow, [1, flow.size()[2]-1], dim=2)
                current_flow = current_flow.squeeze(2)
            else:
                current_flow = calculate_batch_flow(flow_model, current_frame, next_frame)

            current_flow = normalize_optical_flow(current_flow)

            # forward pass
            if count == 1:
                L_rec, L_reg, bpd, mask_prediction = model.forward(current_mask, current_flow, current_frame, next_mask)
            else:
                L_rec_increment, L_reg_increment, bpd_increment, mask_prediction = model.forward(current_mask, current_flow, current_frame, next_mask)
                L_rec = L_rec + L_rec_increment * 1./count
                L_reg = L_reg + L_reg_increment * 1./count
                bpd = bpd + bpd_increment * 1./count

            # update timestep
            count += 1
            current_mask = mask_prediction
        
        # keep a running average
        current = step + 1
        prev_weight = step / current
        average_bpd = average_bpd * prev_weight + bpd / current
        average_rec_loss = average_rec_loss * prev_weight + L_rec / current
        average_reg_loss = average_reg_loss * prev_weight + L_reg / current  

    output = {"performance": average_bpd,
              "BPD": average_bpd, 
              "Reconstruction Loss": average_rec_loss, 
              "Regularization Loss": average_reg_loss, 
              "Negative ELBO": average_rec_loss + average_reg_loss}

    return output


def train_vae(model, extra_models, train_loader, optimizer):
    """
    Function for training a model on a dataset. Train the model for one epoch.
    Inputs:
        model - VAE model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """
    flow_model = extra_models[0]

    # initialize padder object for RAFT
    sample_frame, _, _ = next(iter(train_loader))
    raft_padder = InputPadder(sample_frame.size()[1:])
    del sample_frame

    average_bpd = 0
    average_rec_loss = 0
    average_reg_loss = 0

    for step, batch in enumerate(train_loader):
        # getting batch
        frames, masks, flow = [input.to(model.device) for input in batch]

        current_mask, masks = torch.split(masks, [1, masks.size()[2]-1], dim=2)
        current_mask = raft_padder.pad(current_mask.squeeze(2))[0]

        count = 1
        while frames.size()[2] > 1:
            
            current_frame, frames = torch.split(frames, [1, frames.size()[2]-1], dim=2)
            next_mask, masks = torch.split(masks, [1, masks.size()[2]-1], dim=2)

            current_frame = raft_padder.pad(current_frame.squeeze(2))[0]
            next_mask = raft_padder.pad(next_mask.squeeze(2))[0]

            # prepare optical flow
            if flow is not None:
                current_flow, flow = torch.split(flow, [1, flow.size()[2]-1], dim=2)
                current_flow = current_flow.squeeze(2)
            else:
                current_flow = calculate_batch_flow(flow_model, current_frame, next_frame)

            current_flow = normalize_optical_flow(current_flow)

            # forward pass
            if count == 1:
                L_rec, L_reg, bpd, mask_prediction = model.forward(current_mask, current_flow, current_frame, next_mask)
            else:
                L_rec_increment, L_reg_increment, bpd_increment, mask_prediction = model.forward(current_mask, current_flow, current_frame, next_mask)
                L_rec = L_rec + L_rec_increment * 1./count
                L_reg = L_reg + L_reg_increment * 1./count
                bpd = bpd + bpd_increment * 1./count

            # update timestep
            count += 1
            current_mask = mask_prediction

        # update model
        model.zero_grad()
        bpd.backward()
        optimizer.step()

        # keep a running average
        current = step + 1
        prev_weight = step / current
        average_bpd = average_bpd * prev_weight + bpd / current
        average_rec_loss = average_rec_loss * prev_weight + L_rec / current
        average_reg_loss = average_reg_loss * prev_weight + L_reg / current  

    output = {"performance": average_bpd,
              "BPD": average_bpd, 
              "Reconstruction Loss": average_rec_loss, 
              "Regularization Loss": average_reg_loss, 
              "Negative ELBO": average_rec_loss + average_reg_loss}

    return output


@torch.no_grad()
def demo_model(data_dir, demo_dir, model, extra_models, epoch):
    """
    Run the mask propagation on a full video and create a demo video of the results.

    Args:
        args (namespace): Namespace containing information on the directories
        demo_dir (str): directory in which the results will be saved
        model (nn.Module): Mask propagation VAE that is being trained
    """
    video_dir = os.path.join(data_dir, "JPEGImages/480p/tennis/")
    flow_dir = os.path.join(data_dir, "Flow/480p/flo/forward/tennis/")
    initial_mask = os.path.join(data_dir, "Annotations/480p/tennis/00000.png")

    out_dir = os.path.join(demo_dir, f"epoch_{epoch}/")
    os.makedirs(out_dir, exist_ok=True)

    propagate_mask_through_video(model, video_dir, flow_dir, out_dir, initial_mask)

    masked_vid = create_masked_video(video_dir, out_dir, save_path=os.path.join(demo_dir, f"epoch_{epoch}.mp4"))

    print("####################################################")