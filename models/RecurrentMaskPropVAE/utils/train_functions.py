import os
import numpy as np
import torch

from mask_propagation import propagate_mask_through_video
from utils.flow_utils import normalize_optical_flow
from utils.video_utils import create_masked_video


@torch.no_grad()
def test_vae(model, data_loader):
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
    average_bpd = 0
    average_rec_loss = 0
    average_reg_loss = 0

    for step, batch in enumerate(data_loader):
        # getting batch
        frames, masks, flow = [input.to(model.device) for input in batch]
            
        # prepare optical flow
        if flow is None:
            raise NotImplementedError
        
        flow = normalize_optical_flow(flow)

        # forward pass
        L_rec, L_reg, bpd = model.forward(masks, flow, frames)
        
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


def train_vae(model, data_loader, optimizer):
    """
    Function for training a model on a dataset. Train the model for one epoch.
    Inputs:
        model - VAE model to train
        data_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
    Outputs:
        average_bpd - Average BPD
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """
    average_bpd = 0
    average_rec_loss = 0
    average_reg_loss = 0

    for step, batch in enumerate(data_loader):
        # getting batch
        frames, masks, flow = [input.to(model.device) for input in batch]
            
        # prepare optical flow
        if flow is None:
            raise NotImplementedError

        flow = normalize_optical_flow(flow)

        # forward pass
        L_rec, L_reg, bpd = model.forward(masks, flow, frames)

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
def demo_model(data_dir, demo_dir, model, epoch):
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

    model.encoder.lstm_hidden_state = torch.zeros(1, 1, model.z_dim).to(model.device)
    model.encoder.lstm_cell_state = torch.zeros(1, 1, model.z_dim).to(model.device)

    propagate_mask_through_video(model, video_dir, flow_dir, out_dir, initial_mask)

    masked_vid = create_masked_video(video_dir, out_dir, save_path=os.path.join(demo_dir, f"epoch_{epoch}.mp4"))

    print("####################################################")